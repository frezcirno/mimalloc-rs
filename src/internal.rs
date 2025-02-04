use std::ptr::addr_of;

use crate::init::{MI_HEAP_DEFAULT, MI_HEAP_EMPTY};
use crate::page_queue::_mi_bin;
use crate::segment::mi_segment_page_start;
use crate::types::MiPageKind::MiPageSmall;
use crate::types::{mi_assert, mi_assert_expensive, mi_assert_internal};
use crate::types::{
    MiBlock, MiHeap, MiPage, MiPageQueue, MiSegment, MI_SEGMENT_MASK, MI_SEGMENT_SIZE,
};
use crate::MI_SMALL_SIZE_MAX;

/* -----------------------------------------------------------
  Inlined definitions
----------------------------------------------------------- */

// #define MI_INIT4(x)   x,x,x,x
// #define MI_INIT8(x)   MI_INIT4(x),MI_INIT4(x)
// #define MI_INIT16(x)  MI_INIT8(x),MI_INIT8(x)
// #define MI_INIT32(x)  MI_INIT16(x),MI_INIT16(x)
// #define MI_INIT64(x)  MI_INIT32(x),MI_INIT32(x)
// #define MI_INIT128(x) MI_INIT64(x),MI_INIT64(x)
// #define MI_INIT256(x) MI_INIT128(x),MI_INIT128(x)
macro_rules! MI_INIT {
    (@accum (0, $($_es:expr),*) -> ($($body:tt)*))
        => {MI_INIT!(@as_expr [$($body)*])};
    (@accum (1, $($es:expr), *) -> ($($body:tt)*))
        => {MI_INIT!(@accum (0, $($es),*) -> ($($body)* $($es,)*))};
    (@accum (2, $($es:expr), *) -> ($($body:tt)*))
        => {MI_INIT!(@accum (0, $($es),*) -> ($($body)* $($es,)* $($es,)*))};
    (@accum (3, $($es:expr), *) -> ($($body:tt)*))
        => {MI_INIT!(@accum (2, $($es),*) -> ($($body)* $($es,)*))};
    (@accum (4, $($es:expr), *) -> ($($body:tt)*))
        => {MI_INIT!(@accum (2, $($es,)* $($es),*) -> ($($body)*))};
    (@accum (5, $($es:expr), *) -> ($($body:tt)*))
        => {MI_INIT!(@accum (4, $($es),*) -> ($($body)* $($es,)*))};
    (@accum (6, $($es:expr), *) -> ($($body:tt)*))
        => {MI_INIT!(@accum (4, $($es),*) -> ($($body)* $($es,)* $($es,)*))};
    (@accum (7, $($es:expr), *) -> ($($body:tt)*))
        => {MI_INIT!(@accum (4, $($es),*) -> ($($body)* $($es,)* $($es,)* $($es,)*))};
    (@accum (8, $($es:expr), *) -> ($($body:tt)*))
        => {MI_INIT!(@accum (4, $($es,)* $($es),*) -> ($($body)*))};
    (@accum (16, $($es:expr), *) -> ($($body:tt)*))
        => {MI_INIT!(@accum (8, $($es,)* $($es),*) -> ($($body)*))};
    (@accum (32, $($es:expr), *) -> ($($body:tt)*))
        => {MI_INIT!(@accum (16, $($es,)* $($es),*) -> ($($body)*))};
    (@accum (64, $($es:expr), *) -> ($($body:tt)*))
        => {MI_INIT!(@accum (32, $($es,)* $($es),*) -> ($($body)*))};
    (@accum (128, $($es:expr), *) -> ($($body:tt)*))
        => {MI_INIT!(@accum (64, $($es,)* $($es),*) -> ($($body)*))};
    (@accum (129, $($es:expr), *) -> ($($body:tt)*))
        => {MI_INIT!(@accum (64, $($es,)* $($es),*) -> ($($body)* $($es,)*))};
    (@accum (130, $($es:expr), *) -> ($($body:tt)*))
        => {MI_INIT!(@accum (64, $($es,)* $($es),*) -> ($($body)* $($es,)* $($es,)*))};
    (@accum (256, $($es:expr), *) -> ($($body:tt)*))
        => {MI_INIT!(@accum (128, $($es,)* $($es),*) -> ($($body)*))};

    (@as_expr $e:expr) => {$e};

    [$e:expr; $n:tt] => { MI_INIT!(@accum ($n, $e) -> ()) };
}
use libc::pthread_self;
pub(crate) use MI_INIT;

// Align a byte size to a size in _machine words_,
// i.e. byte size == `wsize*sizeof(void*)`.
pub(crate) fn _mi_wsize_from_size(size: usize) -> usize {
    (size + std::mem::size_of::<usize>() - 1) / std::mem::size_of::<usize>()
}

pub(crate) fn mi_get_default_heap() -> *mut MiHeap {
    MI_HEAP_DEFAULT.with(|heap| heap.get())
}

#[inline]
pub(crate) fn mi_heap_is_default(heap: *const MiHeap) -> bool {
    heap == mi_get_default_heap()
}

#[inline]
pub(crate) fn mi_heap_is_backing(heap: *const MiHeap) -> bool {
    unsafe { (*((*heap).tld)).heap_backing as *const _ == heap }
}

#[inline]
pub(crate) fn mi_heap_is_initialized(heap: *const MiHeap) -> bool {
    heap != addr_of!(MI_HEAP_EMPTY)
}

#[inline]
pub(crate) fn _mi_heap_get_free_small_page(heap: *mut MiHeap, size: usize) -> *mut MiPage {
    mi_assert_internal(size <= MI_SMALL_SIZE_MAX);
    unsafe { (*heap).pages_free_direct[_mi_wsize_from_size(size)] }
}

// Get the page belonging to a certain size class
#[inline]
pub(crate) fn mi_get_free_small_page(size: usize) -> *mut MiPage {
    _mi_heap_get_free_small_page(mi_get_default_heap(), size)
}

// Segment that contains the page or block pointer
#[inline]
pub(crate) fn _mi_ptr_segment(p: *const MiBlock) -> *const MiSegment {
    mi_assert_internal(!p.is_null());
    // 取指针前面的 MI_SEGMENT_MASK 位
    (p as usize & !MI_SEGMENT_MASK) as _
}

// Segment belonging to a page
#[inline]
pub(crate) unsafe fn _mi_page_segment(page: *const MiPage) -> *const MiSegment {
    let segment = _mi_ptr_segment(page as _);
    mi_assert_internal(page == (*segment).pages().offset((*page).segment_idx as _));
    segment
}

// Get the page containing the pointer
#[inline]
pub(crate) unsafe fn _mi_segment_page_of(
    segment: *const MiSegment,
    p: *const MiBlock,
) -> *const MiPage {
    // if (segment->page_size > MI_SEGMENT_SIZE) return &segment->pages[0];  // huge pages
    mi_assert_internal(p as usize >= segment as usize);
    let diff = (p as usize) - (segment as usize);
    mi_assert_internal(diff < MI_SEGMENT_SIZE);
    let idx = diff >> (*segment).page_shift;
    mi_assert_internal(idx < (*segment).capacity);
    mi_assert_internal((*segment).page_kind == MiPageSmall || idx == 0);
    (*segment).pages().offset(idx as _)
}

// Quick page start for initialized pages
#[inline]
pub(crate) unsafe fn _mi_page_start(
    segment: *const MiSegment,
    page: *const MiPage,
    page_size: *mut usize,
) -> *mut u8 {
    mi_segment_page_start(segment, page, page_size)
}

// Get the page containing the pointer
#[inline]
pub(crate) unsafe fn mi_ptr_page(p: *const MiBlock) -> *const MiPage {
    _mi_segment_page_of(_mi_ptr_segment(p), p)
}

// are all blocks in a page freed?
#[inline]
pub(crate) fn mi_page_all_free(page: *const MiPage) -> bool {
    mi_assert_internal(!page.is_null());
    unsafe {
        (*page).used
            - (*page)
                .thread_freed
                .load(std::sync::atomic::Ordering::Relaxed)
            == 0
    }
}

// are there immediately available blocks
#[inline]
pub(crate) fn mi_page_immediate_available(page: *const MiPage) -> bool {
    mi_assert_internal(!page.is_null());
    unsafe { !(*page).free.is_null() }
}

// are there free blocks in this page?
#[inline]
pub(crate) unsafe fn mi_page_has_free(page: *mut MiPage) -> bool {
    mi_assert_internal(!page.is_null());
    let hasfree = unsafe {
        mi_page_immediate_available(page)
            || !(*page).local_free.is_null()
            || (*page).thread_free.head() != 0
    };
    mi_assert_internal(
        hasfree
            || (*page).used
                - (*page)
                    .thread_freed
                    .load(std::sync::atomic::Ordering::Relaxed)
                == (*page).capacity as _,
    );
    hasfree
}

// are all blocks in use?
#[inline]
pub(crate) unsafe fn mi_page_all_used(page: *mut MiPage) -> bool {
    mi_assert_internal(!page.is_null());
    !mi_page_has_free(page)
}

// is more than 7/8th of a page in use?
#[inline]
pub(crate) unsafe fn mi_page_mostly_used(page: *const MiPage) -> bool {
    if page.is_null() {
        return true;
    }
    let frac = (*page).reserved / 8;
    (*page).reserved as usize - (*page).used as usize
        + (*page)
            .thread_freed
            .load(std::sync::atomic::Ordering::Relaxed)
        < frac as usize
}

#[inline]
pub(crate) unsafe fn mi_page_queue(heap: *const MiHeap, size: usize) -> *const MiPageQueue {
    &(*heap).pages[_mi_bin(size) as usize]
}

// -------------------------------------------------------------------
// Encoding/Decoding the free list next pointers
// -------------------------------------------------------------------

#[inline]
pub(crate) unsafe fn mi_block_nextx(block: *const MiBlock) -> *const MiBlock {
    (*block).next as *mut _
}

#[inline]
pub(crate) unsafe fn mi_block_set_nextx(block: *mut MiBlock, next: *const MiBlock) {
    block.write_unaligned(MiBlock { next: next as _ });
}

#[inline]
pub(crate) unsafe fn mi_block_next(page: *const MiPage, block: *const MiBlock) -> *const MiBlock {
    mi_block_nextx(block)
}

#[inline]
pub(crate) unsafe fn mi_block_set_next(
    page: *const MiPage,
    block: *mut MiBlock,
    next: *mut MiBlock,
) {
    mi_block_set_nextx(block, next)
}

// -------------------------------------------------------------------
// Getting the thread id should be performant
// as it is called in the fast path of `_mi_free`,
// so we specialize for various platforms.
// -------------------------------------------------------------------

// TODO

// otherwise use standard C
pub(crate) fn mi_thread_id() -> usize {
    unsafe { pthread_self() as usize }
}
