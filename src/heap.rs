use crate::alloc::mi_heap_malloc;
use crate::init::MI_HEAP_DEFAULT;
use crate::internal::{
    _mi_page_segment, _mi_page_start, _mi_ptr_segment, _mi_segment_page_of, mi_block_next,
    mi_page_all_free,
};
use crate::page::{
    _mi_heap_delayed_free, _mi_page_abandon, _mi_page_free, _mi_page_free_collect,
    mi_deferred_free, mi_page_unfull, mi_page_use_delayed_free,
};
use crate::page_queue::mi_page_queue_append;
use crate::segment::{
    _mi_segment_page_free, _mi_segment_thread_collect, _mi_segment_try_reclaim_abandoned,
};
use crate::types::{MiBlock, MiPage, MiPageQueue, MI_BIN_FULL, MI_INTPTR_SIZE, MI_SMALL_PAGE_SIZE};
use crate::{
    init::{mi_thread_init, MI_HEAP_EMPTY},
    internal::{
        mi_get_default_heap, mi_heap_is_backing, mi_heap_is_default, mi_heap_is_initialized,
        mi_thread_id,
    },
    mi_free,
    types::MiHeap,
};
use crate::types::{mi_assert, mi_assert_expensive, mi_assert_internal};
use crate::{mi_heap_malloc_tp, MiBlockVisitFun, MiHeapArea};
use std::ptr::{addr_of_mut, null_mut};
use std::sync::atomic::Ordering;

/* -----------------------------------------------------------
  Helpers
----------------------------------------------------------- */

pub(crate) type HeapPageVisitorFun = fn(
    heap: *mut MiHeap,
    pq: *mut MiPageQueue,
    page: *mut MiPage,
    arg1: *mut u8,
    arg2: *mut u8,
) -> bool;

unsafe fn mi_heap_visit_pages(
    heap: *mut MiHeap,
    visitor_fn: HeapPageVisitorFun,
    arg1: *mut u8,
    arg2: *mut u8,
) -> bool {
    if heap.is_null() || (*heap).page_count == 0 {
        return false;
    }

    for i in 0..=MI_BIN_FULL as usize {
        let pq = &mut (*heap).pages[i];
        let mut page = pq.first;

        while !page.is_null() {
            let next_page = (*page).next; // Save `next` in case the page gets removed
            assert_eq!((*page).heap as *const _, heap as *const _); // Ensure page belongs to heap

            if !visitor_fn(heap, pq, page, arg1, arg2) {
                return false;
            }

            page = next_page; // Move to the next page
        }
    }

    true
}

fn _mi_heap_page_is_valid(
    heap: *mut MiHeap,
    _pq: *mut MiPageQueue,
    page: *mut MiPage,
    _arg1: *mut u8,
    _arg2: *mut u8,
) -> bool {
    // UNUSED!(arg1);
    // UNUSED!(arg2);
    // UNUSED!(pq);
    unsafe {
        mi_assert_internal({ (*page).heap } == heap);
        let segment = _mi_page_segment(page);
        mi_assert_internal({ (*segment).thread_id } == { (*heap).thread_id });
    }
    true
}

unsafe fn mi_heap_is_valid(heap: *mut MiHeap) -> bool {
    mi_assert_internal(!heap.is_null());
    mi_heap_visit_pages(
        heap,
        _mi_heap_page_is_valid,
        null_mut(),
        null_mut(),
    );
    true
}

/* -----------------------------------------------------------
  "Collect" pages by migrating `local_free` and `thread_free`
  lists and freeing empty pages. This is done when a thread
  stops (and in that case abandons pages if there are still
  blocks alive)
----------------------------------------------------------- */

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd)]
enum MiCollect {
    Normal,
    Force,
    Abandon,
}

fn mi_heap_page_collect(
    _heap: *mut MiHeap,
    pq: *mut MiPageQueue,
    page: *mut MiPage,
    arg_collect: *mut u8,
    _arg2: *mut u8,
) -> bool {
    let collect = unsafe { *(arg_collect as *const MiCollect) };
    unsafe {
        _mi_page_free_collect(page);

        if mi_page_all_free(page) {
            // Free the page if all blocks are free
            _mi_page_free(page, pq, collect != MiCollect::Normal);
        } else if collect == MiCollect::Abandon {
            // Abandon the page if still in use but thread is done
            _mi_page_abandon(page, pq);
        }
    }

    true // Don't break
}

unsafe fn mi_heap_collect_ex(heap: *mut MiHeap, collect: MiCollect) {
    mi_deferred_free(heap, collect > MiCollect::Normal);

    if !mi_heap_is_initialized(heap) {
        return;
    }

    // collect (some) abandoned pages
    if collect >= MiCollect::Normal && !(*heap).no_reclaim {
        if collect == MiCollect::Normal {
            _mi_segment_try_reclaim_abandoned(heap, false, &mut (*(*heap).tld).segments);
        }
    }

    // Mark all full pages to no longer add to delayed_free if abandoning
    if collect == MiCollect::Abandon {
        let mut page: *mut MiPage = (*heap).pages[MI_BIN_FULL as usize].first;
        while !page.is_null() {
            mi_page_use_delayed_free(page, false);
            page = (*page).next;
        }
    }

    // free thread delayed blocks.
    // (if abandoning, after this there are no more local references into the pages.)
    _mi_heap_delayed_free(heap);

    // collect all pages owned by this thread
    mi_heap_visit_pages(
        heap,
        // mi_heap_page_collect as HeapPageVisitorFun,
        unsafe {
            std::mem::transmute(
                mi_heap_page_collect
                    as fn(*mut MiHeap, *mut MiPageQueue, *mut MiPage, *mut u8, *mut u8) -> bool,
            )
        },
        &collect as *const _ as *mut u8,
        null_mut() as *mut u8,
    );
    mi_assert_internal(
        collect != MiCollect::Abandon
            || (*heap)
                .thread_delayed_free
                .load(std::sync::atomic::Ordering::Relaxed)
                .is_null()
    );

    // collect segment caches
    if collect >= MiCollect::Force {
        _mi_segment_thread_collect(&mut (*(*heap).tld).segments);
    }
}

pub(crate) unsafe fn mi_heap_collect_abandon(heap: *mut MiHeap) {
    mi_heap_collect_ex(heap, MiCollect::Abandon);
}

unsafe fn mi_heap_collect(heap: *mut MiHeap, force: bool) {
    mi_heap_collect_ex(
        heap,
        if force {
            MiCollect::Force
        } else {
            MiCollect::Normal
        },
    );
}

unsafe fn mi_collect(force: bool) {
    mi_heap_collect(mi_get_default_heap(), force);
}

/* -----------------------------------------------------------
  Heap new
----------------------------------------------------------- */
pub(crate) fn mi_heap_get_default() -> *mut MiHeap {
    mi_thread_init();
    mi_get_default_heap()
}

fn mi_heap_get_backing() -> *mut MiHeap {
    let heap = mi_heap_get_default();
    mi_assert_internal(!heap.is_null());
    let bheap = unsafe { (*(*heap).tld).heap_backing };
    mi_assert_internal(!bheap.is_null());
    mi_assert_internal(unsafe { (*bheap).thread_id } == mi_thread_id());
    bheap
}

fn mi_heap_new() -> *mut MiHeap {
    let bheap = mi_heap_get_backing();
    let heap = unsafe { mi_heap_malloc_tp!(bheap, MiHeap) };
    if heap.is_null() {
        return null_mut();
    }
    unsafe {
        std::ptr::copy_nonoverlapping(&MI_HEAP_EMPTY as *const _ as *mut _, heap, 1);
        (*heap).tld = (*bheap).tld;
        (*heap).thread_id = mi_thread_id();
        (*heap).no_reclaim = true; // don't reclaim abandoned pages or otherwise destroy is unsafe
    }
    heap
}

// zero out the page queues
unsafe fn mi_heap_reset_pages(heap: *mut MiHeap) {
    mi_assert_internal(mi_heap_is_initialized(heap));
    // TODO: copy full empty heap instead?
    std::ptr::write_volatile(&mut (*heap).pages_free_direct, std::mem::zeroed());
    // #ifdef MI_MEDIUM_DIRECT
    //   memset(&heap->pages_free_medium, 0, sizeof(heap->pages_free_medium));
    // #endif
    std::ptr::copy_nonoverlapping(&MI_HEAP_EMPTY.pages, &mut (*heap).pages, 1);
    (*heap)
        .thread_delayed_free
        .store(null_mut(), Ordering::Relaxed);
    (*heap).page_count = 0;
}

// called from `mi_heap_destroy` and `mi_heap_delete` to free the internal heap resources.
unsafe fn mi_heap_free(heap: *mut MiHeap) {
    mi_assert_internal(mi_heap_is_initialized(heap));
    if mi_heap_is_backing(heap) {
        return; // dont free the backing heap
    }

    // reset default
    if mi_heap_is_default(heap) {
        mi_heap_set_default((*(*heap).tld).heap_backing);
    }

    // and free the used memory
    mi_free(heap as *mut u8);
}

/* -----------------------------------------------------------
  Heap destroy
----------------------------------------------------------- */
fn _mi_heap_page_destroy(
    heap: *mut MiHeap,
    pq: *mut MiPageQueue,
    page: *mut MiPage,
    arg1: *mut u8,
    arg2: *mut u8,
) -> bool {
    // UNUSED macros replaced with Rust-style ignore variables
    let _ = heap;
    let _ = pq;
    let _ = arg1;
    let _ = arg2;

    unsafe {
        // Ensure no more thread_delayed_free will be added
        mi_page_use_delayed_free(page, false);

        // Pretend it is all free now
        mi_assert_internal(
            (*page)
                .thread_freed
                .load(std::sync::atomic::Ordering::Relaxed)
                <= 0xFFFF
        );
        (*page).used = (*page).thread_freed.load(Ordering::Relaxed);

        // Free the page
        _mi_segment_page_free(
            page,
            false, /* no force? */
            addr_of_mut!((*(*heap).tld).segments),
        );
    }

    true // Keep going
}

unsafe fn mi_heap_destroy_pages(heap: *mut MiHeap) {
    mi_heap_visit_pages(
        heap,
        _mi_heap_page_destroy,
        null_mut(),
        null_mut(),
    );
    mi_heap_reset_pages(heap);
}

fn mi_heap_destroy(heap: *mut MiHeap) {
    unsafe {
        mi_assert(mi_heap_is_initialized(heap));
        mi_assert((*heap).no_reclaim);
        mi_assert_expensive(mi_heap_is_valid(heap));

        if !mi_heap_is_initialized(heap) {
            return;
        }

        if !(*heap).no_reclaim {
            // Don't free in case it may contain reclaimed pages
            mi_heap_delete(heap);
        } else {
            // Free all pages
            mi_heap_destroy_pages(heap);
            mi_heap_free(heap);
        }
    }
}

/* -----------------------------------------------------------
  Safe Heap delete
----------------------------------------------------------- */
// Transfer the pages from one heap to the other
fn mi_heap_absorb(heap: *mut MiHeap, from: *mut MiHeap) {
    unsafe {
        mi_assert_internal(!heap.is_null());
        if from.is_null() || (*from).page_count == 0 {
            return;
        }

        // Unfull all full pages
        let mut page = (*heap).pages[MI_BIN_FULL as usize].first;
        while !page.is_null() {
            let next = (*page).next;
            mi_page_unfull(page);
            page = next;
        }
        mi_assert_internal((*heap).pages[MI_BIN_FULL as usize].first.is_null());

        // Free outstanding thread-delayed free blocks
        _mi_heap_delayed_free(from);

        // Transfer all pages by appending the queues
        for i in 0..MI_BIN_FULL as usize {
            let pq = &mut (*heap).pages[i];
            let append = &mut (*from).pages[i];
            mi_page_queue_append(heap, pq, append);
        }
        mi_assert_internal((*from)
            .thread_delayed_free
            .load(Ordering::Relaxed)
            .is_null());

        // Reset the `from` heap
        mi_heap_reset_pages(from);
    }
}

// Safely delete a heap without freeing any still allocated blocks in that heap
fn mi_heap_delete(heap: *mut MiHeap) {
    unsafe {
        mi_assert(mi_heap_is_initialized(heap));
        mi_assert_expensive(mi_heap_is_valid(heap));
        if !mi_heap_is_initialized(heap) {
            return;
        }

        if !mi_heap_is_backing(heap) {
            // Transfer still-used pages to the backing heap
            mi_heap_absorb((*(*heap).tld).heap_backing, heap);
        } else {
            // The backing heap abandons its pages
            mi_heap_collect_abandon(heap);
        }
        mi_assert_internal((*heap).page_count == 0);
        mi_heap_free(heap);
    }
}

pub(crate) unsafe fn mi_heap_set_default(heap: *mut MiHeap) -> *mut MiHeap {
    mi_assert(mi_heap_is_initialized(heap));
    mi_assert_expensive(mi_heap_is_valid(heap));
    let old = MI_HEAP_DEFAULT.get();
    MI_HEAP_DEFAULT.set(heap);
    old
}

/* -----------------------------------------------------------
  Analysis
----------------------------------------------------------- */
// Static since it is not thread-safe to access heaps from other threads
fn mi_heap_of_block(p: *const MiBlock) -> *mut MiHeap {
    if p.is_null() {
        return null_mut();
    }
    unsafe {
        let segment = _mi_ptr_segment(p);
        (*_mi_segment_page_of(segment, p)).heap
    }
}

fn mi_heap_contains_block(heap: *mut MiHeap, p: *const MiBlock) -> bool {
    mi_assert(!heap.is_null());
    if !mi_heap_is_initialized(heap) {
        return false;
    }
    heap == mi_heap_of_block(p)
}

fn mi_heap_page_check_owned(
    heap: *mut MiHeap,
    pq: *mut MiPageQueue,
    page: *mut MiPage,
    p: *mut u8,
    vfound: *mut u8,
) -> bool {
    // UNUSED macros replaced with Rust-style ignore variables
    let _ = heap;
    let _ = pq;

    unsafe {
        let found = &mut *(vfound as *mut bool);
        let segment = _mi_page_segment(page);
        let start = _mi_page_start(segment, page, null_mut());
        let end = (start as *mut u8).add((*page).capacity as usize * (*page).block_size) as *mut u8;
        *found = p >= start && p < end;
        !*found // Continue if not found
    }
}

fn mi_heap_check_owned(heap: *mut MiHeap, p: *const MiBlock) -> bool {
    unsafe {
        mi_assert(!heap.is_null());
        if !mi_heap_is_initialized(heap) {
            return false;
        }
        if (p as usize & (MI_INTPTR_SIZE - 1)) != 0 {
            return false; // Only aligned pointers
        }
        let mut found = false;
        mi_heap_visit_pages(
            heap,
            mi_heap_page_check_owned,
            p as *mut _,
            &mut found as *const _ as *mut _,
        );
        found
    }
}

fn mi_check_owned(p: *const MiBlock) -> bool {
    mi_heap_check_owned(mi_get_default_heap(), p)
}

/* -----------------------------------------------------------
  Visit all heap blocks and areas
  Todo: enable visiting abandoned pages, and
        enable visiting all blocks of all heaps across threads
----------------------------------------------------------- */
// Separate struct to keep `MiPage` out of the public interface
#[repr(C)]
struct MiHeapAreaEx {
    area: MiHeapArea,
    page: *mut MiPage,
}

fn mi_heap_area_visit_blocks(
    xarea: *mut MiHeapAreaEx,
    visitor: *mut MiBlockVisitFun,
    arg: *mut u8,
) -> bool {
    unsafe {
        mi_assert(!xarea.is_null());
        if xarea.is_null() {
            return true;
        }

        let area = &(*xarea).area;
        let page = (*xarea).page;
        mi_assert(!page.is_null());
        if page.is_null() {
            return true;
        }

        _mi_page_free_collect(page);
        mi_assert_internal((*page).local_free.is_null());
        if (*page).used == 0 {
            return true;
        }

        let pstart = _mi_page_start(_mi_page_segment(page), page, null_mut());

        if (*page).capacity == 1 {
            // Optimize page with one block
            mi_assert_internal((*page).used == 1 && (*page).free.is_null());
            return (*visitor)(
                (*page).heap,
                area as *const _ as *mut _,
                pstart,
                (*page).block_size,
                arg,
            );
        }

        // Create a bitmap of free blocks
        const MI_MAX_BLOCKS: usize = MI_SMALL_PAGE_SIZE / std::mem::size_of::<*mut u8>();
        let mut free_map: [usize; MI_MAX_BLOCKS / std::mem::size_of::<usize>()] =
            [0; MI_MAX_BLOCKS / std::mem::size_of::<usize>()];

        let mut free_count = 0;
        let mut block = (*page).free;
        while !block.is_null() {
            free_count += 1;
            let offset = (block as *const u8).offset_from(pstart as *const u8) as usize;
            let blockidx = offset / (*page).block_size;
            let bitidx = blockidx / std::mem::size_of::<usize>();
            let bit = blockidx % std::mem::size_of::<usize>();
            free_map[bitidx] |= 1 << bit;

            block = mi_block_next(page, block) as _;
        }

        mi_assert_internal((*page).capacity as usize == free_count + (*page).used);

        // Walk through all blocks, skipping the free ones
        let mut used_count = 0;
        for i in 0..(*page).capacity as usize {
            let bitidx = i / std::mem::size_of::<usize>();
            let bit = i % std::mem::size_of::<usize>();
            if (free_map[bitidx] & (1 << bit)) == 0 {
                used_count += 1;
                let block = pstart.add(i * (*page).block_size);
                if !(*visitor)(
                    (*page).heap,
                    area as *const _ as *mut _,
                    block,
                    (*page).block_size,
                    arg,
                ) {
                    return false;
                }
            }
        }
        mi_assert_internal((*page).used == used_count);
        true
    }
}

type MiHeapAreaVisitFun = fn(heap: *mut MiHeap, area: *mut MiHeapAreaEx, arg: *mut u8) -> bool;

fn mi_heap_visit_areas_page(
    heap: *mut MiHeap,
    pq: *mut MiPageQueue,
    page: *mut MiPage,
    vfun: *mut u8,
    arg: *mut u8,
) -> bool {
    unsafe {
        let _ = heap;
        let _ = pq;

        let fun = std::mem::transmute::<_, MiHeapAreaVisitFun>(vfun);
        let mut xarea = MiHeapAreaEx {
            area: MiHeapArea {
                reserved: (*page).reserved as usize * (*page).block_size,
                committed: (*page).capacity as usize * (*page).block_size,
                blocks: {
                    let mut page_size_unused = 0;
                    _mi_page_start(_mi_page_segment(page), page, &mut page_size_unused)
                },
                used: (*page).used - (*page).thread_freed.load(Ordering::Relaxed), // Race condition is acceptable
                block_size: (*page).block_size,
            },
            page,
        };

        (fun)(heap, addr_of_mut!(xarea), arg)
    }
}

fn mi_heap_visit_areas(heap: *mut MiHeap, visitor: *mut MiHeapAreaVisitFun, arg: *mut u8) -> bool {
    if visitor.is_null() {
        return false;
    }

    unsafe {
        mi_heap_visit_pages(
            heap as *mut MiHeap,
            mi_heap_visit_areas_page,
            visitor as *mut _,
            arg,
        )
    }
}
// Just to pass arguments
#[repr(C)]
struct MiVisitBlocksArgs {
    visit_blocks: bool,
    visitor: *mut MiBlockVisitFun,
    arg: *mut u8,
}

fn mi_heap_area_visitor(heap: *mut MiHeap, xarea: *mut MiHeapAreaEx, arg: *mut u8) -> bool {
    unsafe {
        let args = &mut *(arg as *mut MiVisitBlocksArgs);
        if !(*args.visitor)(
            heap,
            addr_of_mut!((*xarea).area),
            null_mut(),
            (*xarea).area.block_size,
            args.arg,
        ) {
            return false;
        }
        if args.visit_blocks {
            mi_heap_area_visit_blocks(xarea, args.visitor, args.arg)
        } else {
            true
        }
    }
}

// Visit all blocks in a heap
fn mi_heap_visit_blocks(
    heap: *mut MiHeap,
    visit_blocks: bool,
    visitor: *mut MiBlockVisitFun,
    arg: *mut u8,
) -> bool {
    let mut args = MiVisitBlocksArgs {
        visit_blocks,
        visitor,
        arg,
    };

    mi_heap_visit_areas(
        heap,
        // addr_of_mut!(mi_heap_area_visitor),
        unsafe {
            std::mem::transmute(
                mi_heap_area_visitor as fn(*mut MiHeap, *mut MiHeapAreaEx, *mut u8) -> bool,
            )
        },
        &mut args as *mut _ as *mut u8,
    )
}
