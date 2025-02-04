use crate::init::{MI_HEAP_EMPTY, MI_PAGE_EMPTY};
use crate::internal::_mi_wsize_from_size;
use crate::os::{mi_align_up, mi_os_page_size};
use crate::types::{mi_assert, mi_assert_internal};
use crate::types::{
    MiHeap, MiPage, MiPageQueue, MI_BIN_FULL, MI_BIN_HUGE, MI_LARGE_SIZE_MAX, MI_LARGE_WSIZE_MAX,
};
use crate::MI_SMALL_SIZE_MAX;
use std::ptr::{addr_of_mut, null_mut};

/* -----------------------------------------------------------
  Queue query
----------------------------------------------------------- */

pub(crate) fn mi_page_queue_is_huge(pq: *const MiPageQueue) -> bool {
    unsafe { (*pq).block_size == (MI_LARGE_SIZE_MAX + std::mem::size_of::<usize>()) }
}

pub(crate) fn mi_page_queue_is_full(pq: *const MiPageQueue) -> bool {
    unsafe { (*pq).block_size == (MI_LARGE_SIZE_MAX + (2 * std::mem::size_of::<usize>())) }
}

pub(crate) fn mi_page_queue_is_special(pq: *const MiPageQueue) -> bool {
    unsafe { (*pq).block_size > MI_LARGE_SIZE_MAX }
}

/* -----------------------------------------------------------
  Bins
----------------------------------------------------------- */

// Bit scan reverse: return the index of the highest bit.
#[inline]
pub(crate) fn _mi_bsr(x: usize) -> u8 {
    (usize::BITS - 1 - x.leading_zeros()) as _
}

#[inline]
pub(crate) fn mi_bsr32(x: u32) -> u8 {
    (u32::BITS - 1 - x.leading_zeros()) as _
}

// Return the bin for a given field size.
// Returns MI_BIN_HUGE if the size is too large.
// We use `wsize` for the size in "machine word sizes",
// i.e. byte size == `wsize*sizeof(void*)`.
#[inline]
pub(crate) fn _mi_bin(size: usize) -> u8 {
    let mut wsize = _mi_wsize_from_size(size);
    let bin: u8;
    if wsize <= 1 {
        bin = 1;
    } else if wsize <= 8 {
        bin = wsize as u8;
    } else if wsize > MI_LARGE_WSIZE_MAX {
        bin = MI_BIN_HUGE;
    } else {
        wsize -= 1;
        // find the highest bit
        let b = mi_bsr32(wsize as u32);
        // and use the top 3 bits to determine the bin (~16% worst internal fragmentation).
        // - adjust with 3 because we use do not round the first 8 sizes
        //   which each get an exact bin
        bin = ((b << 2) + ((wsize >> (b - 2)) & 0x03) as u8) - 3;
    }
    mi_assert_internal(bin > 0 && bin <= MI_BIN_HUGE);
    bin
}

/* -----------------------------------------------------------
  Queue of pages with free blocks
----------------------------------------------------------- */

pub(crate) fn mi_bin_size(bin: u8) -> usize {
    MI_HEAP_EMPTY.pages[bin as usize].block_size
}

// Good size for allocation
pub(crate) fn mi_good_size(size: usize) -> usize {
    if size <= MI_LARGE_SIZE_MAX {
        return mi_bin_size(_mi_bin(size));
    } else {
        return mi_align_up(size, mi_os_page_size());
    }
}

pub(crate) unsafe fn mi_page_queue_of(page: *const MiPage) -> *const MiPageQueue {
    let bin = if (*page).flags.in_full() {
        MI_BIN_FULL
    } else {
        _mi_bin((*page).block_size)
    };
    let heap = (*page).heap;
    mi_assert_internal(!heap.is_null() && bin <= MI_BIN_FULL);
    let pq = &mut (*heap).pages[bin as usize];
    mi_assert_internal(bin >= MI_BIN_HUGE || (*page).block_size == (*pq).block_size);
    pq
}

pub(crate) unsafe fn mi_heap_page_queue_of(
    heap: *const MiHeap,
    page: *const MiPage,
) -> *const MiPageQueue {
    let bin = if (*page).flags.in_full() {
        MI_BIN_FULL
    } else {
        _mi_bin((*page).block_size)
    };
    mi_assert_internal(bin <= MI_BIN_FULL);
    let pq = &(*heap).pages[bin as usize];
    mi_assert_internal((*page).flags.in_full() || (*page).block_size == (*pq).block_size);
    pq
}

// The current small page array is for efficiency and for each
// small size (up to 256) it points directly to the page for that
// size without having to compute the bin. This means when the
// current free page queue is updated for a small bin, we need to update a
// range of entries in `_mi_page_small_free`.
pub(crate) unsafe fn mi_heap_queue_first_update(heap: *mut MiHeap, pq: *const MiPageQueue) {
    let size = (*pq).block_size;
    if size > MI_SMALL_SIZE_MAX {
        return;
    }
    let mut page = (*pq).first;
    if (*pq).first.is_null() {
        page = addr_of_mut!(MI_PAGE_EMPTY);
    }

    // find index in the right direct page array
    let mut start: usize;
    let idx = _mi_wsize_from_size(size);
    let pages_free = &mut (*heap).pages_free_direct;

    if pages_free[idx] == page {
        return; // already set
    }

    // find start slot
    if idx <= 1 {
        start = 0;
    } else {
        // find previous size; due to minimal alignment upto 3 previous bins may need to be skipped
        let bin = _mi_bin(size);
        let mut prev: *mut MiPageQueue = pq.sub(1) as *mut _;
        while bin == _mi_bin((*prev).block_size) && prev > addr_of_mut!((*heap).pages[0]) {
            prev = prev.wrapping_sub(1);
        }
        start = 1 + _mi_wsize_from_size((*prev).block_size);
        if start > idx {
            start = idx;
        }
    }

    // set size range to the right page
    mi_assert(start <= idx);
    for sz in start..=idx {
        pages_free[sz] = page;
    }
}

/*
static bool mi_page_queue_is_empty(mi_page_queue_t* queue) {
  return (queue->first == NULL);
}
*/

pub(crate) unsafe fn mi_page_queue_remove(queue: *mut MiPageQueue, page: *mut MiPage) {
    mi_assert_internal(!page.is_null());
    mi_assert_internal(
        (*page).block_size == (*queue).block_size
            || ((*page).block_size > MI_LARGE_SIZE_MAX && mi_page_queue_is_huge(queue))
            || ((*page).flags.in_full() && mi_page_queue_is_full(queue)),
    );
    if !(*page).prev.is_null() {
        (*(*page).prev).next = (*page).next;
    }
    if !(*page).next.is_null() {
        (*(*page).next).prev = (*page).prev;
    }
    if page == (*queue).last {
        (*queue).last = (*page).prev;
    }
    if page == (*queue).first {
        (*queue).first = (*page).next;
        // update first
        let heap = (*page).heap;
        mi_heap_queue_first_update(heap, queue);
    }
    (*(*page).heap).page_count -= 1;
    (*page).next = null_mut();
    (*page).prev = null_mut();
    (*page).heap = null_mut();
    (*page).flags.set_in_full(false);
}

// push a page to the front of a queue
pub(crate) unsafe fn mi_page_queue_push(
    heap: *mut MiHeap,
    queue: *mut MiPageQueue,
    page: *mut MiPage,
) {
    mi_assert_internal((*page).heap.is_null());
    mi_assert_internal(
        (*page).block_size == (*queue).block_size
            || ((*page).block_size > MI_LARGE_SIZE_MAX && mi_page_queue_is_huge(queue))
            || ((*page).flags.in_full() && mi_page_queue_is_full(queue)),
    );
    (*page).flags.set_in_full(mi_page_queue_is_full(queue));
    (*page).heap = heap;
    (*page).next = (*queue).first;
    (*page).prev = null_mut();
    if !(*queue).first.is_null() {
        mi_assert_internal((*(*queue).first).prev.is_null());
        (*(*queue).first).prev = page;
        (*queue).first = page;
    } else {
        (*queue).first = page;
        (*queue).last = page;
    }

    // update direct
    mi_heap_queue_first_update(heap, queue);
    (*heap).page_count += 1;
}

pub(crate) unsafe fn mi_page_queue_enqueue_from(
    to: *mut MiPageQueue,
    from: *mut MiPageQueue,
    page: *mut MiPage,
) {
    mi_assert_internal(!page.is_null());
    mi_assert_internal(
        (*page).block_size == (*to).block_size
            || ((*page).block_size > MI_LARGE_SIZE_MAX && mi_page_queue_is_huge(to))
            || ((*page).block_size == (*from).block_size && mi_page_queue_is_full(to)),
    );

    if !(*page).prev.is_null() {
        (*(*page).prev).next = (*page).next;
    }
    if !(*page).next.is_null() {
        (*(*page).next).prev = (*page).prev;
    }
    if page == (*from).last {
        (*from).last = (*page).prev;
    }
    if page == (*from).first {
        (*from).first = (*page).next;
        // update first
        let heap = (*page).heap;
        mi_heap_queue_first_update(heap, from);
    }

    (*page).prev = (*to).last;
    (*page).next = null_mut();
    if !(*to).last.is_null() {
        mi_assert_internal((*page).heap == (*(*to).last).heap);
        (*(*to).last).next = page;
        (*to).last = page;
    } else {
        (*to).first = page;
        (*to).last = page;
        mi_heap_queue_first_update((*page).heap, to);
    }

    (*page).flags.set_in_full(mi_page_queue_is_full(to));
}

pub(crate) unsafe fn mi_page_queue_append(
    heap: *mut MiHeap,
    pq: *mut MiPageQueue,
    append: *mut MiPageQueue,
) {
    mi_assert_internal((*pq).block_size == (*append).block_size);

    if (*append).first.is_null() {
        return;
    }

    // set append pages to new heap
    let mut page = (*append).first;
    while !page.is_null() {
        (*page).heap = heap;
        page = (*page).next;
    }

    if (*pq).last.is_null() {
        // take over afresh
        mi_assert_internal((*pq).first.is_null());
        (*pq).first = (*append).first;
        (*pq).last = (*append).last;
        mi_heap_queue_first_update(heap, pq);
    } else {
        // append to end
        mi_assert_internal(!(*pq).last.is_null());
        mi_assert_internal(!(*append).first.is_null());
        (*(*pq).last).next = (*append).first;
        (*(*append).first).prev = (*pq).last;
        (*pq).last = (*append).last;
    }
}
