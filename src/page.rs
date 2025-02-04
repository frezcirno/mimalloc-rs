use crate::alloc::{_mi_page_malloc, mi_free_delayed_block};
use crate::init::mi_thread_init;
use crate::internal::{
    _mi_page_segment, _mi_page_start, _mi_wsize_from_size, mi_block_next, mi_block_nextx,
    mi_block_set_next, mi_get_default_heap, mi_heap_is_initialized, mi_page_all_free,
    mi_page_immediate_available, mi_page_mostly_used, mi_page_queue,
};
use crate::page_queue::mi_page_queue_enqueue_from;
use crate::page_queue::{
    _mi_bin, mi_heap_page_queue_of, mi_page_queue_is_huge, mi_page_queue_of, mi_page_queue_push,
    mi_page_queue_remove,
};
use crate::segment::_mi_segment_try_reclaim_abandoned;
use crate::segment::{_mi_segment_page_alloc, mi_segment_page_start};
use crate::segment::{_mi_segment_page_free, mi_segment_page_abandon};
use crate::types::{mi_assert, mi_assert_internal};
use crate::types::{MiBlock, MiPage, MiPageQueue, MiThreadFree, MI_BIN_HUGE};
use crate::types::{MiDelayed::*, MiHeap, MI_BIN_FULL, MI_LARGE_SIZE_MAX, MI_TF_PTR_SHIFT};
use std::ptr::null_mut;
use std::sync::atomic::Ordering;

/* -----------------------------------------------------------
  Page helpers
----------------------------------------------------------- */

// Index a block in a page
#[inline]
pub(crate) unsafe fn mi_page_block_at(
    page: *const MiPage,
    page_start: *mut u8,
    i: usize,
) -> *mut MiBlock {
    debug_assert!(!page.is_null());
    debug_assert!(i <= (*page).reserved as usize);
    unsafe { page_start.add(i * (*page).block_size) as *mut MiBlock }
}

pub(crate) unsafe fn mi_page_use_delayed_free(page: *mut MiPage, enable: bool) {
    let mut tfree: MiThreadFree;
    let mut tfreex: MiThreadFree;

    loop {
        tfreex = (*page).thread_free.clone();
        tfree = (*page).thread_free.clone();
        tfreex.set_delayed(if enable {
            MiUseDelayedFree
        } else {
            MiNoDelayedFree
        });
        if tfree.delayed() == MiDelayedFreeing {
            std::hint::spin_loop(); // Yield and retry, until outstanding MI_DELAYED_FREEING are done.
            continue;
        }

        if tfreex.delayed() == tfree.delayed() {
            break;
        }

        if !(*page)
            .thread_free
            .compare_exchange(tfreex.load(), tfree.load())
            .is_ok()
        {
            continue;
        }
        break;
    }
}

/* -----------------------------------------------------------
  Page collect the `local_free` and `thread_free` lists
----------------------------------------------------------- */

// Collect the local `thread_free` list using an atomic exchange.
// Note: The exchange must be done atomically as this is used right after
// moving to the full list in `mi_page_collect_ex` and we need to
// ensure that there was no race where the page became unfull just before the move.
pub(crate) unsafe fn mi_page_thread_free_collect(page: *mut MiPage) {
    let mut head: *mut MiBlock;
    let mut tfree: MiThreadFree;
    let mut tfreex: MiThreadFree;

    loop {
        tfreex = (*page).thread_free.clone();
        tfree = (*page).thread_free.clone();
        head = ((tfree.head() as usize) << MI_TF_PTR_SHIFT) as usize as *mut MiBlock;
        tfreex.set_head(0);
        if (*page)
            .thread_free
            .compare_exchange(tfreex.load(), tfree.load())
            .is_ok()
        {
            break;
        }
    }

    // Return if the list is empty
    if head.is_null() {
        return;
    }

    // Find the tail
    let mut count = 1;
    let mut tail = head;

    loop {
        tail = mi_block_next(page, tail) as _;
        if tail.is_null() {
            break;
        }
        count += 1;
    }

    // Prepend to the free list
    mi_block_set_next(page, tail, (*page).free);
    (*page).free = head;

    // Update counts now
    (*page).thread_freed.fetch_sub(count, Ordering::Relaxed);
    (*page).used -= count;
}

pub(crate) unsafe fn _mi_page_free_collect(page: *mut MiPage) {
    mi_assert_internal(!page.is_null());

    // Free the local free list
    if !(*page).local_free.is_null() {
        if (*page).free.is_null() {
            // Usual case
            (*page).free = (*page).local_free;
        } else {
            let mut tail = (*page).free;
            let mut next: *mut MiBlock;

            loop {
                next = mi_block_next(page, tail) as _;
                if next.is_null() {
                    break;
                }
                tail = next;
            }

            mi_block_set_next(page, tail, (*page).local_free);
        }
        (*page).local_free = null_mut();
    }

    // And the thread free list
    if (*page).thread_free.head() != 0 {
        // Quick test to avoid an atomic operation
        mi_page_thread_free_collect(page);
    }
}

/* -----------------------------------------------------------
  Page fresh and retire
----------------------------------------------------------- */
// Called from segments when reclaiming abandoned pages
pub(crate) unsafe fn _mi_page_reclaim(heap: *mut MiHeap, page: *mut MiPage) {
    mi_assert_internal((*page).heap.is_null());
    _mi_page_free_collect(page);
    let pq = mi_page_queue(heap, (*page).block_size);
    mi_page_queue_push(heap, pq as *mut _, page);
}

// Allocate a fresh page from a segment
unsafe fn mi_page_fresh_alloc(
    heap: *mut MiHeap,
    pq: *mut MiPageQueue,
    block_size: usize,
) -> *mut MiPage {
    let page = _mi_segment_page_alloc(
        block_size,
        &mut (*(*heap).tld).segments,
        &mut (*(*heap).tld).os,
    );
    if page.is_null() {
        return null_mut();
    }
    mi_page_init(heap, page, block_size);
    mi_page_queue_push(heap, pq, page);
    page
}

// Get a fresh page to use
unsafe fn mi_page_fresh(heap: *mut MiHeap, pq: *mut MiPageQueue) -> *mut MiPage {
    // Try to reclaim an abandoned page first
    let mut page = (*pq).first;
    if !(*heap).no_reclaim
        && _mi_segment_try_reclaim_abandoned(heap, false, &mut (*(*heap).tld).segments)
        && page != (*pq).first
    {
        // We reclaimed, and we got lucky with a reclaimed page in our queue
        page = (*pq).first;
        if !(*page).free.is_null() {
            return page;
        }
    }
    // Otherwise, allocate the page
    page = mi_page_fresh_alloc(heap, pq, (*pq).block_size);
    if page.is_null() {
        return null_mut();
    }
    assert_eq!((*pq).block_size, (*page).block_size);
    assert_eq!(pq, mi_page_queue(heap, (*page).block_size) as _);
    page
}

/* -----------------------------------------------------------
   Do any delayed frees
   (put there by other threads if they deallocated in a full page)
----------------------------------------------------------- */
pub(crate) unsafe fn _mi_heap_delayed_free(heap: *mut MiHeap) {
    // Take over the thread-delayed free list and free all blocks
    let mut block: *mut MiBlock;
    loop {
        block = (*heap).thread_delayed_free.load(Ordering::Relaxed);
        if block.is_null()
            || (*heap)
                .thread_delayed_free
                .compare_exchange(
                    block as *mut _,
                    null_mut(),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
        {
            break;
        }
    }

    // Free all blocks
    while !block.is_null() {
        let next = mi_block_nextx(block);
        mi_free_delayed_block(block);
        block = next as _;
    }
}

/* -----------------------------------------------------------
  Unfull, abandon, free and retire
----------------------------------------------------------- */

// Move a page from the full list back to a regular list
pub(crate) unsafe fn mi_page_unfull(page: *mut MiPage) {
    mi_assert_internal(!page.is_null());
    mi_assert_internal((*page).flags.in_full());

    mi_page_use_delayed_free(page, false);
    if !(*page).flags.in_full() {
        return;
    }

    let heap = (*page).heap;
    let pqfull = &mut (*heap).pages[MI_BIN_FULL as usize];
    (*page).flags.set_in_full(false);
    let pq = mi_heap_page_queue_of(heap, page);
    (*page).flags.set_in_full(true);
    mi_page_queue_enqueue_from(pq as *mut _, pqfull, page);
}

// Move a page to the full list
unsafe fn mi_page_to_full(page: *mut MiPage, pq: *mut MiPageQueue) {
    mi_assert_internal(pq == mi_page_queue_of(page) as *mut _);
    mi_assert_internal(!mi_page_immediate_available(page));
    mi_assert_internal(!(*page).flags.in_full());

    mi_page_use_delayed_free(page, true);
    if (*page).flags.in_full() {
        return;
    }

    mi_page_queue_enqueue_from(&mut (*(*page).heap).pages[MI_BIN_FULL as usize], pq, page);
    mi_page_thread_free_collect(page);
}

// Abandon a page with used blocks at the end of a thread.
// Note: only call if it is ensured that no references exist from
// the `page->heap->thread_delayed_free` into this (*page).
// Currently only called through `mi_heap_collect_ex` which ensures this.
pub(crate) unsafe fn _mi_page_abandon(page: *mut MiPage, pq: *mut MiPageQueue) {
    mi_assert_internal(!page.is_null());
    mi_assert_internal(pq == mi_page_queue_of(page) as *mut _);
    mi_assert_internal(!(*page).heap.is_null());
    mi_assert_internal((*page).thread_free.delayed() == MiNoDelayedFree);

    let segments_tld = &mut (*(*(*page).heap).tld).segments;
    mi_page_queue_remove(pq, page);
    mi_segment_page_abandon(page, segments_tld);
}

// Free a page with no more free blocks
pub(crate) unsafe fn _mi_page_free(page: *mut MiPage, pq: *mut MiPageQueue, force: bool) {
    mi_assert_internal(!page.is_null());
    mi_assert_internal(pq == mi_page_queue_of(page) as *mut _);
    mi_assert_internal(mi_page_all_free(page));
    mi_assert_internal((*page).thread_free.delayed() != MiDelayedFreeing);

    (*page).flags.set_has_aligned(false);

    // account for huge pages here

    // remove from the page list
    // (no need to do _mi_heap_delayed_free first as all blocks are already free)
    let segments_tld = &mut (*(*(*page).heap).tld).segments;
    mi_page_queue_remove(pq, page);

    // and free it
    mi_assert_internal((*page).heap.is_null());
    _mi_segment_page_free(page, force, segments_tld);
}

// Retire a page with no more used blocks
// Important to not retire too quickly though as new
// allocations might coming.(*(*page).heap).tld
// Note: called from `mi_free` and benchmarks often
// trigger this due to freeing everything and then
// allocating again so careful when changing this.
pub(crate) unsafe fn mi_page_retire(page: *mut MiPage) {
    mi_assert_internal(!page.is_null());
    mi_assert_internal(mi_page_all_free(page));

    (*page).flags.set_has_aligned(false);

    // don't retire too often..
    // (or we end up retiring and re-allocating most of the time)
    // NOTE: refine this more: we should not retire if this
    // is the only page left with free blocks. It is not clear
    // how to check this efficiently though... for now we just check
    // if its neighbours are almost fully used.
    // if mi_likely((*page).block_size <= MI_LARGE_SIZE_MAX) {
    if (*page).block_size <= MI_LARGE_SIZE_MAX {
        if mi_page_mostly_used((*page).prev) && mi_page_mostly_used((*page).next) {
            return;
        }
    }

    _mi_page_free(page, mi_page_queue_of(page) as *mut _, false);
}

/* -----------------------------------------------------------
  Initialize the initial free list in a (*page).
  In secure mode we initialize a randomized list by
  alternating between slices.
----------------------------------------------------------- */

const MI_MAX_SLICE_SHIFT: usize = 6; // At most 64 slices
const MI_MAX_SLICES: usize = 1 << MI_MAX_SLICE_SHIFT;
const MI_MIN_SLICES: usize = 2;

pub(crate) unsafe fn mi_page_free_list_extend(
    _heap: *mut MiHeap,
    page: *mut MiPage,
    extend: usize,
) {
    let page_area = unsafe { _mi_page_start(_mi_page_segment(page), page, null_mut()) };
    let bsize = (*page).block_size;
    let start = unsafe { mi_page_block_at(page, page_area, (*page).capacity.into()) };

    // Initialize a sequential free list
    let end = unsafe {
        mi_page_block_at(
            page,
            page_area,
            ((*page).capacity as usize + extend - 1).into(),
        )
    };
    let mut block = start;

    for _ in 0..extend {
        let next = unsafe { (block as *mut u8).add(bsize) as *mut MiBlock };
        unsafe { mi_block_set_next(page, block, next) };
        block = next;
    }

    unsafe { mi_block_set_next(page, end, null_mut()) };
    (*page).free = start;
    (*page).capacity += extend as u16;
}

const MI_MAX_EXTEND_SIZE: usize = 4 * 1024;
const MI_MIN_EXTEND: usize = 1;

// Extend the capacity (up to reserved) by initializing a free list
// We do at most `MI_MAX_EXTEND` to avoid touching too much memory
// Note: we also experimented with "bump" allocation on the first
// allocations but this did not speed up any benchmark (due to an
// extra test in malloc? or cache effects?)
pub(crate) unsafe fn mi_page_extend_free(heap: *mut MiHeap, page: *mut MiPage) {
    mi_assert((*page).free.is_null());
    mi_assert((*page).local_free.is_null());
    // mi_assert_expensive(mi_page_is_valid_init(page));
    if (*page).capacity >= (*page).reserved {
        return;
    }

    _mi_page_start(_mi_page_segment(page), page, null_mut());

    if (*page).is_reset {
        (*page).is_reset = false;
    }

    let mut extend: usize = (*page).reserved as usize - (*page).capacity as usize as usize;
    let max_extend = MI_MAX_EXTEND_SIZE / (*page).block_size;

    let max_extend = if max_extend < MI_MIN_EXTEND {
        MI_MIN_EXTEND
    } else {
        max_extend
    };

    if extend > max_extend {
        extend = max_extend.max(1);
    }

    mi_page_free_list_extend(heap, page, extend);
}

// Initialize a fresh page
unsafe fn mi_page_init(heap: *mut MiHeap, page: *mut MiPage, block_size: usize) {
    mi_assert(!page.is_null());
    let segment = _mi_page_segment(page);
    mi_assert(!segment.is_null());

    // set fields
    let mut page_size = 0;
    mi_segment_page_start(segment, page, &mut page_size);
    (*page).block_size = block_size;
    mi_assert_internal(block_size > 0);
    mi_assert_internal(page_size / block_size < (1 << 16));
    (*page).reserved = (page_size / block_size) as u16;

    mi_assert_internal((*page).capacity == 0);
    mi_assert_internal((*page).free.is_null());
    mi_assert_internal((*page).used == 0);
    mi_assert_internal((*page).thread_free.load() == 0);
    mi_assert_internal((*page).thread_freed.load(Ordering::Relaxed) == 0);
    mi_assert_internal((*page).next.is_null());
    mi_assert_internal((*page).prev.is_null());
    mi_assert_internal(!(*page).flags.has_aligned());
    // mi_assert_expensive(mi_page_is_valid_init(page));

    // initialize an initial free list
    mi_page_extend_free(heap, page);
    mi_assert(mi_page_immediate_available(page));
}

/* -----------------------------------------------------------
  Find pages with free blocks
-------------------------------------------------------------*/

// Find a page with free blocks of `page->block_size`.
pub(crate) unsafe fn mi_page_queue_find_free_ex(
    heap: *mut MiHeap,
    pq: *mut MiPageQueue,
) -> *mut MiPage {
    let mut rpage: *mut MiPage = null_mut();
    let mut page_free_count = 0;
    let mut page = (*pq).first;

    while !page.is_null() {
        // 0. collect freed blocks by us and other threads
        _mi_page_free_collect(page);

        // 1. if the page contains free blocks, we are done
        if mi_page_immediate_available(page) {
            // If all blocks are free, we might retire this page instead.
            // do this at most 8 times to bound allocation time.
            // (note: this can happen if a page was earlier not retired due
            //  to having neighbours that were mostly full or due to concurrent frees)
            if page_free_count < 8 && mi_page_all_free(page) {
                page_free_count += 1;
                if !rpage.is_null() {
                    _mi_page_free(rpage, pq, false);
                }
                rpage = page;
                page = (*page).next;
                continue; // Keep looking
            } else {
                break; // Use this page
            }
        }

        // 2. Try to extend
        if (*page).capacity < (*page).reserved {
            mi_page_extend_free(heap, page);
            debug_assert!(mi_page_immediate_available(page));
            break;
        }

        // 3. If the page is completely full, move it to the `mi_pages_full`
        // queue so we don't visit long-lived pages too often.
        mi_assert_internal(!(*page).flags.in_full() && !mi_page_immediate_available(page));
        mi_page_to_full(page, pq);

        page = (*page).next;
    } // for each page

    if page.is_null() {
        page = rpage;
        rpage = null_mut();
    };
    if !rpage.is_null() {
        _mi_page_free(rpage, pq, false);
    }

    if page.is_null() {
        page = mi_page_fresh(heap, pq);
    } else {
        mi_assert((*pq).first == page);
    }
    mi_assert_internal(mi_page_immediate_available(page));
    page
}

pub(crate) unsafe fn mi_find_free_page(heap: *mut MiHeap, size: usize) -> *mut MiPage {
    _mi_heap_delayed_free(heap);
    let pq = mi_page_queue(heap, size);
    let page = (*pq).first;
    if !page.is_null() {
        _mi_page_free_collect(page);
        if mi_page_immediate_available(page) {
            return page; // fast path
        }
    }
    mi_page_queue_find_free_ex(heap, pq as *mut _)
}

/* -----------------------------------------------------------
  Users can register a deferred free function called
  when the `free` list is empty. Since the `local_free`
  is separate this is deterministically called after
  a certain number of allocations.
----------------------------------------------------------- */

type MiDeferredFreeFun = fn(force: bool, heartbeat: usize);

static mut DEFERRED_FREE: Option<MiDeferredFreeFun> = None;

pub(crate) unsafe fn mi_deferred_free(heap: *mut MiHeap, force: bool) {
    (*(*heap).tld).heartbeat.fetch_add(1, Ordering::SeqCst); // Increment the heartbeat
    if let Some(deferred_free) = DEFERRED_FREE {
        deferred_free(force, (*(*heap).tld).heartbeat.load(Ordering::SeqCst) as _);
    }
}

pub(crate) unsafe fn mi_register_deferred_free(fn_ptr: MiDeferredFreeFun) {
    DEFERRED_FREE = Some(fn_ptr);
}

/* -----------------------------------------------------------
  General allocation
----------------------------------------------------------- */

// A huge page is allocated directly without being in a queue
unsafe fn mi_huge_page_alloc(heap: *mut MiHeap, size: usize) -> *mut MiPage {
    let block_size = _mi_wsize_from_size(size) * std::mem::size_of::<usize>();
    mi_assert_internal(_mi_bin(block_size) == MI_BIN_HUGE);
    let pq = mi_page_queue(heap, block_size);
    mi_assert_internal(mi_page_queue_is_huge(pq));
    let page = mi_page_fresh_alloc(heap, pq as *mut _, block_size);
    if !page.is_null() {
        mi_assert_internal(mi_page_immediate_available(page));
        mi_assert_internal(unsafe { (*page).block_size } == block_size);
    }
    page
}

// Generic allocation routine if the fast path (`alloc.c:mi_page_malloc`) does not succeed.
pub(crate) unsafe fn _mi_malloc_generic(mut heap: *mut MiHeap, size: usize) -> *mut u8 {
    mi_assert_internal(!heap.is_null());

    // initialize if necessary
    if !mi_heap_is_initialized(heap) {
        mi_thread_init(); // calls `_mi_heap_init` in turn
        heap = mi_get_default_heap();
    }
    mi_assert_internal(mi_heap_is_initialized(heap));

    // call potential deferred free routines
    mi_deferred_free(heap, false);

    // huge allocation?
    let page: *mut MiPage;
    if size > MI_LARGE_SIZE_MAX {
        // Note: unlikely
        page = mi_huge_page_alloc(heap, size);
    } else {
        // otherwise find a page with free blocks in our size segregated queues
        page = mi_find_free_page(heap, size);
    }

    if page.is_null() {
        return null_mut(); // out of memory
    }

    mi_assert_internal(mi_page_immediate_available(page));
    mi_assert_internal(unsafe { (*page).block_size } >= size);

    // and try again, this time succeeding! (i.e. this should never recurse)
    _mi_page_malloc(heap, page, size)
}
