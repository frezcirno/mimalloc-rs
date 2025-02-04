use crate::internal::{
    _mi_heap_get_free_small_page, _mi_page_start, _mi_ptr_segment, _mi_segment_page_of,
    mi_block_next, mi_block_set_next, mi_block_set_nextx, mi_get_default_heap, mi_page_all_free,
    mi_ptr_page, mi_thread_id,
};
use crate::page::{_mi_malloc_generic, mi_page_retire, mi_page_unfull};
use crate::types::{MiBlock, MiDelayed, MiHeap, MiPage, MiSegment, MI_TF_PTR_SHIFT};
use crate::MI_SMALL_SIZE_MAX;
use crate::types::{mi_assert, mi_assert_internal};
use std::{ptr::null_mut, sync::atomic::Ordering};

// ------------------------------------------------------
// Allocation
// ------------------------------------------------------

// Fast allocation in a page: just pop from the free list.
// Fall back to generic allocation only if the list is empty.
pub(crate) unsafe fn _mi_page_malloc(heap: *mut MiHeap, page: *mut MiPage, size: usize) -> *mut u8 {
    mi_assert_internal((*page).block_size == 0 || (*page).block_size >= size);
    let block = (*page).free;
    if block.is_null() {
        // slow path, the page is unused
        return _mi_malloc_generic(heap, size);
    }
    mi_assert_internal(!block.is_null() && mi_ptr_page(block) == page);

    // pop from the free list
    (*page).free = mi_block_next(page, block) as _;
    (*page).used += 1;
    mi_assert_internal((*page).free.is_null() || mi_ptr_page((*page).free) == page);

    block as *mut u8
}

// allocate a small block
fn mi_heap_malloc_small(heap: *mut MiHeap, size: usize) -> *mut u8 {
    mi_assert(size <= MI_SMALL_SIZE_MAX);
    let page = _mi_heap_get_free_small_page(heap, size);
    unsafe { _mi_page_malloc(heap, page, size) }
}

#[inline]
unsafe fn mi_malloc_small(size: usize) -> *mut u8 {
    mi_heap_malloc_small(mi_get_default_heap(), size)
}

// zero initialized small block
#[inline]
unsafe fn mi_zalloc_small(size: usize) -> *mut u8 {
    let p = mi_malloc_small(size);
    if !p.is_null() {
        std::ptr::write_bytes(p as *mut u8, 0, size);
    }
    p
}

// 主分配函数
#[inline]
pub unsafe fn mi_heap_malloc(heap: *mut MiHeap, size: usize) -> *mut u8 {
    mi_assert(!heap.is_null());
    mi_assert((*heap).thread_id == 0 || (*heap).thread_id == mi_thread_id()); // 堆是线程本地的
    let p: *mut u8;
    if size <= MI_SMALL_SIZE_MAX {
        p = mi_heap_malloc_small(heap, size);
    } else {
        p = _mi_malloc_generic(heap, size);
    }
    p
}

#[inline]
pub unsafe fn mi_malloc(size: usize) -> *mut u8 {
    mi_heap_malloc(mi_get_default_heap(), size)
}

pub unsafe fn _mi_heap_malloc_zero(heap: *mut MiHeap, size: usize, zero: bool) -> *mut u8 {
    let p = mi_heap_malloc(heap, size);
    if zero && !p.is_null() {
        std::ptr::write_bytes(p, 0, size);
    }
    p
}

#[inline]
pub unsafe fn mi_heap_zalloc(heap: *mut MiHeap, size: usize) -> *mut u8 {
    _mi_heap_malloc_zero(heap, size, true)
}

pub unsafe fn mi_zalloc(size: usize) -> *mut u8 {
    mi_heap_zalloc(mi_get_default_heap(), size)
}

// ------------------------------------------------------
// Free
// ------------------------------------------------------

// multi-threaded free
unsafe fn mi_free_block_mt(page: *mut MiPage, block: *mut MiBlock) {
    let mut tfree;
    let mut tfreex;
    let mut use_delayed;

    loop {
        tfreex = (*page).thread_free.clone();
        tfree = (*page).thread_free.clone();
        use_delayed = tfree.delayed() == MiDelayed::MiUseDelayedFree;
        if use_delayed {
            // unlikely: first concurrent free in a full list page
            tfreex.set_delayed(MiDelayed::MiDelayedFreeing);
        } else {
            // usual: directly add to the thread_free list
            mi_block_set_next(
                page,
                block,
                (tfree.head() << MI_TF_PTR_SHIFT) as *mut MiBlock,
            );
            tfreex.set_head((block as usize >> MI_TF_PTR_SHIFT) as usize);
        }

        if (*page)
            .thread_free
            .compare_exchange(tfree.load(), tfreex.load())
            .is_ok()
        {
            break;
        }
    }

    if !use_delayed {
        // increment thread_free count and return
        (*page).thread_freed.fetch_add(1, Ordering::Release);
    } else {
        // racy read on `heap`, but acceptable because MI_DELAYED_FREEING is set
        let heap = (*page).heap;
        mi_assert_internal(!heap.is_null());
        if !heap.is_null() {
            // add to the delayed free list of this heap
            let mut dfree;
            loop {
                dfree = (*heap).thread_delayed_free.load(Ordering::Relaxed);
                mi_block_set_nextx(block, dfree);
                if (*heap)
                    .thread_delayed_free
                    .compare_exchange(dfree, block, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
                {
                    break;
                }
            }
        }

        // reset MI_DELAYED_FREEING flag
        loop {
            tfreex = tfree.clone();
            tfreex.set_delayed(MiDelayed::MiNoDelayedFree);
            if (*page)
                .thread_free
                .compare_exchange(tfree.load(), tfreex.load())
                .is_ok()
            {
                break;
            }
            tfree = (*page).thread_free.clone();
        }
    }
}

// regular free
#[inline]
unsafe fn mi_free_block(page: *mut MiPage, local: bool, block: *mut MiBlock) {
    if local {
        // owning thread can free a block directly
        mi_block_set_next(page, block, (*page).local_free);
        (*page).local_free = block;
        (*page).used -= 1;
        if mi_page_all_free(page) {
            mi_page_retire(page);
        } else if (*page).flags.in_full() {
            mi_page_unfull(page);
        }
    } else {
        mi_free_block_mt(page, block);
    }
}

// Adjust a block that was allocated aligned, to the actual start of the block in the (*page).
unsafe fn _mi_page_ptr_unalign(
    segment: *const MiSegment,
    page: *const MiPage,
    p: *const MiBlock,
) -> *const MiBlock {
    mi_assert_internal(!page.is_null() && !p.is_null());
    let diff = p.offset_from(_mi_page_start(segment, page, null_mut()) as _) as usize;
    let adjust = diff % (*page).block_size;
    p.sub(adjust)
}

unsafe fn mi_free_generic(segment: *mut MiSegment, page: *mut MiPage, local: bool, p: *mut u8) {
    let block = if (*page).flags.has_aligned() {
        _mi_page_ptr_unalign(segment, page, p as *mut _).cast_mut()
    } else {
        p as _
    };
    mi_free_block(page, local, block);
}

// Free a block
pub unsafe fn mi_free(p: *mut u8) {
    if p.is_null() {
        return;
    }

    let segment = _mi_ptr_segment(p as _);
    if segment.is_null() {
        return; // checks for p == NULL
    }
    let local = mi_thread_id() == (*segment).thread_id;

    let page = _mi_segment_page_of(segment, p as _);

    if (*page).flags.0 == 0 {
        let block = p as *mut MiBlock;
        if local {
            // owning thread can free a block directly
            mi_block_set_next(page, block, (*page).local_free);
            (*page.cast_mut()).local_free = block;
            (*page.cast_mut()).used -= 1;
            if mi_page_all_free(page) {
                mi_page_retire(page.cast_mut());
            }
        } else {
            // use atomic operations for a multi-threaded free
            mi_free_block_mt(page.cast_mut(), block);
        }
    } else {
        // aligned blocks, or a full page; use the more generic path
        mi_free_generic(segment.cast_mut(), page.cast_mut(), local, p);
    }
}

pub unsafe fn mi_free_delayed_block(block: *mut MiBlock) {
    mi_assert_internal(!block.is_null());
    let segment = _mi_ptr_segment(block);
    debug_assert_eq!(mi_thread_id(), (*segment).thread_id);
    let page = _mi_segment_page_of(segment, block);
    mi_free_block(page.cast_mut(), true, block);
}

// Bytes available in a block
pub unsafe fn mi_usable_size(p: *mut u8) -> usize {
    if p.is_null() {
        return 0;
    }
    let p = p as *mut MiBlock;
    let segment = _mi_ptr_segment(p);
    let page = _mi_segment_page_of(segment, p);
    let size = (*page).block_size;
    if (*page).flags.has_aligned() {
        let adjust = p.offset_from(_mi_page_ptr_unalign(segment, page, p) as *mut _);
        mi_assert_internal(adjust >= 0 && adjust as usize <= size);
        size - adjust as usize
    } else {
        size
    }
}
