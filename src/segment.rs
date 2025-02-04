use std::io::Write;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::{ptr::addr_of_mut, sync::atomic::AtomicUsize};

use crate::os::mi_os_alloc_aligned;
use crate::page::_mi_page_reclaim;
use crate::types::{mi_assert, mi_assert_expensive, mi_assert_internal};
use crate::types::{MiHeap, MI_LARGE_PAGE_SHIFT, MI_SMALL_PAGE_SHIFT};
use crate::{
    internal::{_mi_page_segment, _mi_ptr_segment, mi_page_all_free, mi_ptr_page, mi_thread_id},
    os::{mi_align_up, mi_os_free, mi_os_page_size, mi_os_reset, mi_os_shrink},
    types::{
        MiOsTld, MiPage, MiPageKind, MiSegment, MiSegmentQueue, MiSegmentsTld, MI_LARGE_SIZE_MAX,
        MI_MAX_ALIGN_SIZE, MI_SEGMENT_SHIFT, MI_SEGMENT_SIZE, MI_SMALL_PAGES_PER_SEGMENT,
        MI_SMALL_PAGE_SIZE,
    },
};

const MI_PAGE_HUGH_ALIGN: usize = 256 * 1024;

/* -----------------------------------------------------------
  Segment allocation
  We allocate pages inside big OS allocated "segments"
  (4mb on 64-bit). This is to avoid splitting VMA's on Linux
  and reduce fragmentation on other OS's. Each thread
  owns its own segments.

  Currently we have:
  - small pages (64kb), 32 in one segment
  - large pages (4mb), 1 in one segment
  - huge blocks > MI_LARGE_SIZE_MAX (512kb) are directly allocated by the OS

  In any case the memory for a segment is virtual and only
  committed on demand (i.e. we are careful to not touch the memory
  until we actually allocate a block there)

  If a  thread ends, it "abandons" pages with used blocks
  and there is an abandoned segment list whose segments can
  be reclaimed by still running threads, much like work-stealing.
----------------------------------------------------------- */

unsafe fn mi_segment_is_valid(segment: *const MiSegment) -> bool {
    mi_assert_internal(segment != null_mut());
    mi_assert_internal((*segment).used <= (*segment).capacity);
    mi_assert_internal((*segment).abandoned <= (*segment).used);
    let mut nfree = 0;
    for i in 0..(*segment).capacity {
        if !(*(*segment).pages().offset(i as isize)).segment_in_use {
            nfree += 1;
        }
    }
    mi_assert_internal(nfree + (*segment).used == (*segment).capacity);
    mi_assert_internal((*segment).thread_id == mi_thread_id()); // or 0
    true
}

/* -----------------------------------------------------------
  Queue of segments containing free pages
----------------------------------------------------------- */

unsafe fn mi_segment_queue_contains(queue: *const MiSegmentQueue, segment: *mut MiSegment) -> bool {
    mi_assert_internal(segment != null_mut());
    let mut list = (*queue).first;
    while !list.is_null() {
        if list == segment {
            break;
        }
        mi_assert_internal((*list).next.is_null() || (*(*list).next).prev == list);
        mi_assert_internal((*list).prev.is_null() || (*(*list).prev).next == list);
        list = (*list).next;
    }
    list == segment
}

// quick test to see if a segment is in the free pages queue
unsafe fn mi_segment_is_in_free_queue(segment: *mut MiSegment, tld: *mut MiSegmentsTld) -> bool {
    let in_queue = !(*segment).next.is_null()
        || !(*segment).prev.is_null()
        || (*tld).small_free.first == segment;
    if in_queue {
        mi_assert_internal((*segment).page_kind == MiPageKind::MiPageSmall);
        mi_assert_expensive(mi_segment_queue_contains(
            addr_of_mut!((*tld).small_free),
            segment,
        ));
    }
    in_queue
}

#[inline]
unsafe fn mi_segment_queue_is_empty(queue: *const MiSegmentQueue) -> bool {
    (*queue).first.is_null()
}

unsafe fn mi_segment_queue_remove(queue: *mut MiSegmentQueue, segment: *mut MiSegment) {
    mi_assert_expensive(mi_segment_queue_contains(queue, segment));
    if !(*segment).prev.is_null() {
        (*(*segment).prev).next = (*segment).next;
    }
    if !(*segment).next.is_null() {
        (*(*segment).next).prev = (*segment).prev;
    }
    if segment == (*queue).first {
        (*queue).first = (*segment).next;
    }
    if segment == (*queue).last {
        (*queue).last = (*segment).prev;
    }
    (*segment).next = null_mut();
    (*segment).prev = null_mut();
}

unsafe fn mi_segment_enqueue(queue: *mut MiSegmentQueue, segment: *mut MiSegment) {
    mi_assert_expensive(!mi_segment_queue_contains(queue, segment));
    (*segment).next = null_mut();
    (*segment).prev = (*queue).last;
    if !(*queue).last.is_null() {
        mi_assert_internal((*(*queue).last).next.is_null());
        (*(*queue).last).next = segment;
        (*queue).last = segment;
    } else {
        (*queue).last = segment;
        (*queue).first = segment;
    }
}

unsafe fn mi_segment_queue_insert_before(
    queue: *mut MiSegmentQueue,
    elem: *mut MiSegment,
    segment: *mut MiSegment,
) {
    mi_assert_expensive(elem.is_null() || mi_segment_queue_contains(queue, elem));
    mi_assert_expensive(!segment.is_null() && !mi_segment_queue_contains(queue, segment));
    (*segment).prev = if elem.is_null() {
        (*queue).last
    } else {
        (*elem).prev
    };
    if !(*segment).prev.is_null() {
        (*(*segment).prev).next = segment;
    } else {
        (*queue).first = segment;
    }
    (*segment).next = elem;
    if !(*segment).next.is_null() {
        (*(*segment).next).prev = segment;
    } else {
        (*queue).last = segment;
    }
}

// Start of the page available memory
pub(crate) unsafe fn mi_segment_page_start(
    segment: *const MiSegment,
    page: *const MiPage,
    page_size: *mut usize,
) -> *mut u8 {
    let mut psize = if (*segment).page_kind == MiPageKind::MiPageHuge {
        (*segment).segment_size
    } else {
        1 << (*segment).page_shift
    };
    let mut p = (segment as usize + (*page).segment_idx as usize * psize) as *mut u8;
    if (*page).segment_idx == 0 {
        // the first page starts after the segment info (and possible guard page)
        p = (p as usize + (*segment).segment_info_size) as *mut u8;
        psize -= (*segment).segment_info_size;
    }
    if !page_size.is_null() {
        *page_size = psize;
    }
    mi_assert_internal(mi_ptr_page(p as _) == page as *mut MiPage);
    mi_assert_internal(_mi_ptr_segment(p as _) == segment as *mut MiSegment);
    p
}

unsafe fn mi_segment_size(
    capacity: usize,
    required: usize,
    pre_size: *mut usize,
    info_size: *mut usize,
) -> usize {
    let minsize =
        std::mem::size_of::<MiSegment>() + ((capacity - 1) * std::mem::size_of::<MiPage>()) + 16; /* padding */
    let guardsize = 0;
    let isize_ = mi_align_up(minsize, 16.max(MI_MAX_ALIGN_SIZE));
    *info_size = isize_;
    if !pre_size.is_null() {
        *pre_size = isize_ + guardsize;
    }
    if required == 0 {
        MI_SEGMENT_SIZE
    } else {
        mi_align_up(required + isize_ + 2 * guardsize, MI_PAGE_HUGH_ALIGN)
    }
}

/* -----------------------------------------------------------
Segment caches
We keep a small segment cache per thread to avoid repeated allocation
and free in the OS if a program allocates memory and then frees
all again repeatedly. (We tried a one-element cache but that
proves to be too small for certain workloads).
----------------------------------------------------------- */

unsafe fn mi_segments_track_size(segment_size: isize, tld: *mut MiSegmentsTld) {
    (*tld).current_size = ((*tld).current_size as isize + segment_size) as _;
    if (*tld).current_size > (*tld).peak_size {
        (*tld).peak_size = (*tld).current_size;
    }
}

unsafe fn mi_segment_os_free(
    segment: *mut MiSegment,
    segment_size: usize,
    tld: *mut MiSegmentsTld,
) {
    mi_segments_track_size(-(segment_size as isize), tld);
    mi_os_free(segment as *mut _, segment_size);
}

// The segment cache is limited to be at most 1/8 of the peak size
// in use (and no more than 32)
const MI_SEGMENT_CACHE_MAX: usize = 32;
const MI_SEGMENT_CACHE_FRACTION: usize = 8;

// Get a segment of at least `required` size.
// If `required == MI_SEGMENT_SIZE` the `segment_size` will match exactly
unsafe fn _mi_segment_cache_findx(
    tld: *mut MiSegmentsTld,
    required: usize,
    reverse: bool,
) -> *mut MiSegment {
    mi_assert_internal(required % mi_os_page_size() == 0);
    let mut segment = if reverse {
        (*tld).cache.last
    } else {
        (*tld).cache.first
    };
    while !segment.is_null() {
        if (*segment).segment_size >= required {
            (*tld).cache_count -= 1;
            (*tld).cache_size -= (*segment).segment_size;
            mi_segment_queue_remove(addr_of_mut!((*tld).cache), segment);
            // exact size match?
            if required == 0 || (*segment).segment_size == required {
                return segment;
            }
            // not more than 25% waste and on a huge page segment? (in that case the segment size does not need to match required)
            else if required != MI_SEGMENT_SIZE
                && (*segment).segment_size - ((*segment).segment_size / 4) <= required
            {
                return segment;
            }
            // try to shrink the memory to match exactly
            else {
                if mi_os_shrink(segment as *mut u8, (*segment).segment_size, required) {
                    (*tld).current_size -= (*segment).segment_size;
                    (*tld).current_size += required;
                    (*segment).segment_size = required;
                    return segment;
                } else {
                    // if that all fails, we give up
                    mi_segment_os_free(segment, (*segment).segment_size, tld);
                    return null_mut();
                }
            }
        }
        segment = if reverse {
            (*segment).prev
        } else {
            (*segment).next
        };
    }
    null_mut()
}

#[inline]
unsafe fn mi_segment_cache_find(tld: *mut MiSegmentsTld, required: usize) -> *mut MiSegment {
    _mi_segment_cache_findx(tld, required, false)
}

#[inline]
unsafe fn mi_segment_cache_evict(tld: *mut MiSegmentsTld) -> *mut MiSegment {
    // TODO: random eviction instead?
    _mi_segment_cache_findx(tld, 0, true /* from the end */)
}

unsafe fn mi_segment_cache_full(tld: *mut MiSegmentsTld) -> bool {
    if (*tld).cache_count < MI_SEGMENT_CACHE_MAX
        && (*tld).cache_size * MI_SEGMENT_CACHE_FRACTION < (*tld).peak_size
    {
        return false;
    }
    // take the opportunity to reduce the segment cache if it is too large (now)
    while (*tld).cache_size * MI_SEGMENT_CACHE_FRACTION >= (*tld).peak_size + 1 {
        let segment = mi_segment_cache_evict(tld);
        mi_assert_internal(segment != null_mut());
        if !segment.is_null() {
            mi_segment_os_free(segment, (*segment).segment_size, tld);
        }
    }
    true
}

unsafe fn mi_segment_cache_insert(segment: *mut MiSegment, tld: *mut MiSegmentsTld) -> bool {
    mi_assert_internal((*segment).next.is_null() && (*segment).prev.is_null());
    mi_assert_internal(!mi_segment_is_in_free_queue(segment, tld));
    mi_assert_expensive(!mi_segment_queue_contains(&(*tld).cache, segment));
    if mi_segment_cache_full(tld) {
        return false;
    }
    // if mi_option_is_enabled(mi_option_cache_reset) && !mi_option_is_enabled(mi_option_page_reset) {
    //     mi_os_reset(
    //         (segment as usize + (*segment).segment_info_size) as *mut u8,
    //         (*segment).segment_size - (*segment).segment_info_size,
    //     );
    // }
    // insert ordered
    let mut seg = (*tld).cache.first;
    while !seg.is_null() && (*seg).segment_size < (*segment).segment_size {
        seg = (*seg).next;
    }
    mi_segment_queue_insert_before(addr_of_mut!((*tld).cache), seg, segment);
    (*tld).cache_count += 1;
    (*tld).cache_size += (*segment).segment_size;
    true
}

// called by ending threads to free cached segments
pub(crate) unsafe fn _mi_segment_thread_collect(tld: *mut MiSegmentsTld) {
    let mut segment = mi_segment_cache_find(tld, 0);
    while !segment.is_null() {
        mi_segment_os_free(segment, MI_SEGMENT_SIZE, tld);
        segment = mi_segment_cache_find(tld, 0);
    }
    mi_assert_internal((*tld).cache_count == 0 && (*tld).cache_size == 0);
    mi_assert_internal(mi_segment_queue_is_empty(&(*tld).cache));
}

/* -----------------------------------------------------------
   Segment allocation
----------------------------------------------------------- */

// Allocate a segment from the OS aligned to `MI_SEGMENT_SIZE` .
unsafe fn mi_segment_alloc(
    required: usize,
    page_kind: MiPageKind,
    page_shift: usize,
    tld: *mut MiSegmentsTld,
    os_tld: *mut MiOsTld,
) -> *mut MiSegment {
    // calculate needed sizes first
    let capacity = if page_kind == MiPageKind::MiPageHuge {
        mi_assert_internal(page_shift == MI_SEGMENT_SHIFT && required > 0);
        1
    } else {
        mi_assert_internal(required == 0);
        let page_size = 1 << page_shift;
        let capacity = MI_SEGMENT_SIZE / page_size;
        mi_assert_internal(MI_SEGMENT_SIZE % page_size == 0);
        mi_assert_internal(capacity >= 1 && capacity <= MI_SMALL_PAGES_PER_SEGMENT);
        capacity
    };
    let mut info_size = 0;
    let mut pre_size = 0;
    let segment_size = mi_segment_size(capacity, required, &mut pre_size, &mut info_size);
    mi_assert_internal(segment_size >= required);

    // Allocate the segment
    let mut segment = mi_segment_cache_find(tld, segment_size);
    mi_assert_internal(
        segment.is_null()
            || (segment_size == MI_SEGMENT_SIZE && segment_size == (*segment).segment_size)
            || (segment_size != MI_SEGMENT_SIZE && segment_size <= (*segment).segment_size),
    );

    // and otherwise allocate it from the OS
    if segment.is_null() {
        segment = mi_os_alloc_aligned(segment_size, MI_SEGMENT_SIZE, os_tld) as *mut MiSegment;
        if segment.is_null() {
            return null_mut();
        }
        mi_segments_track_size(segment_size as isize, tld);
    }

    mi_assert_internal((segment as usize) % MI_SEGMENT_SIZE == 0);

    std::ptr::write_bytes(segment, 0, info_size);

    (*segment).page_kind = page_kind;
    (*segment).capacity = capacity;
    (*segment).page_shift = page_shift;
    (*segment).segment_size = segment_size;
    (*segment).segment_info_size = pre_size;
    (*segment).thread_id = mi_thread_id();
    for i in 0..(*segment).capacity {
        (*(*segment).pages().offset(i as _)).segment_idx = i as u8;
    }
    segment
}

// Available memory in a page
unsafe fn mi_page_size(page: *const MiPage) -> usize {
    let mut psize = 0;
    mi_segment_page_start(_mi_page_segment(page), page, &mut psize);
    psize
}

unsafe fn mi_segment_free(segment: *mut MiSegment, force: bool, tld: *mut MiSegmentsTld) {
    mi_assert_internal(segment != null_mut());
    if mi_segment_is_in_free_queue(segment, tld) {
        if (*segment).page_kind != MiPageKind::MiPageSmall {
            eprintln!(
                "mimalloc: expecting small segment: {:?}, {:?}, {:?}, {:?}",
                (*segment).page_kind,
                (*segment).prev,
                (*segment).next,
                (*tld).small_free.first
            );
            assert!(std::io::stderr().flush().is_ok());
        } else {
            mi_assert_internal((*segment).page_kind == MiPageKind::MiPageSmall); // for now we only support small pages
            mi_assert_expensive(mi_segment_queue_contains(
                addr_of_mut!((*tld).small_free),
                segment,
            ));
            mi_segment_queue_remove(addr_of_mut!((*tld).small_free), segment);
        }
    }
    mi_assert_expensive(!mi_segment_queue_contains(
        addr_of_mut!((*tld).small_free),
        segment,
    ));
    mi_assert_internal((*segment).next.is_null());
    mi_assert_internal((*segment).prev.is_null());
    (*segment).thread_id = 0;

    // update reset memory statistics

    if !force && mi_segment_cache_insert(segment, tld) {
        // it is put in our cache
    } else {
        // otherwise return it to the OS
        mi_segment_os_free(segment, (*segment).segment_size, tld);
    }
}

/* -----------------------------------------------------------
  Free page management inside a segment
----------------------------------------------------------- */

#[inline]
unsafe fn mi_segment_has_free(segment: *const MiSegment) -> bool {
    (*segment).used < (*segment).capacity
}

unsafe fn mi_segment_find_free(segment: *mut MiSegment) -> *mut MiPage {
    mi_assert_internal(mi_segment_has_free(segment));
    mi_assert_expensive(mi_segment_is_valid(segment));
    for i in 0..(*segment).capacity {
        let page = (*segment).pages().offset(i as _);
        if !(*page).segment_in_use {
            return page;
        }
    }
    mi_assert(false);
    null_mut()
}

/* -----------------------------------------------------------
    Free
----------------------------------------------------------- */

unsafe fn mi_segment_page_clear(segment: *mut MiSegment, page: *mut MiPage) {
    mi_assert_internal((*page).segment_in_use);
    mi_assert_internal(mi_page_all_free(page));
    let inuse = (*page).capacity as usize * (*page).block_size;

    // reset the page memory to reduce memory pressure?
    if !(*page).is_reset && false
    /*mi_option_is_enabled(mi_option_page_reset)*/
    {
        let mut psize: usize = 0;
        let start = mi_segment_page_start(segment, page, &mut psize);
        (*page).is_reset = true;
        if inuse > 0 {
            mi_os_reset(start, inuse);
        }
    }

    // zero the page data
    let idx = (*page).segment_idx; // don't clear the index
    let is_reset = (*page).is_reset; // don't clear the reset flag
    std::ptr::write_bytes(page, 0, 1);
    (*page).segment_idx = idx;
    (*page).segment_in_use = false;
    (*page).is_reset = is_reset;
    (*segment).used -= 1;
}

pub(crate) unsafe fn _mi_segment_page_free(
    page: *mut MiPage,
    force: bool,
    tld: *mut MiSegmentsTld,
) {
    mi_assert(!page.is_null());
    let segment = _mi_page_segment(page);
    mi_assert_expensive(mi_segment_is_valid(segment));

    // mark it as free now
    mi_segment_page_clear(segment.cast_mut(), page);

    if (*segment).used == 0 {
        // no more used pages; remove from the free list and free the segment
        mi_segment_free(segment.cast_mut(), force, tld);
    } else {
        if (*segment).used == (*segment).abandoned {
            // only abandoned pages; remove from free list and abandon
            mi_segment_abandon(segment.cast_mut(), tld);
        } else if (*segment).used + 1 == (*segment).capacity {
            mi_assert_internal((*segment).page_kind == MiPageKind::MiPageSmall); // for now we only support small pages
                                                                                 // move back to segments small pages free list
            mi_segment_enqueue(addr_of_mut!((*tld).small_free), segment.cast_mut());
        }
    }
}

/* -----------------------------------------------------------
   Abandonment
----------------------------------------------------------- */

// When threads terminate, they can leave segments with
// live blocks (reached through other threads). Such segments
// are "abandoned" and will be reclaimed by other threads to
// reuse their pages and/or free them eventually
static mut ABANDONED: AtomicPtr<MiSegment> = AtomicPtr::new(null_mut());
static mut ABANDONED_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe fn mi_segment_abandon(segment: *mut MiSegment, tld: *mut MiSegmentsTld) {
    mi_assert_internal((*segment).used == (*segment).abandoned);
    mi_assert_internal((*segment).used > 0);
    mi_assert_internal((*segment).abandoned_next.is_null());
    mi_assert_expensive(mi_segment_is_valid(segment));

    // remove the segment from the free page queue if needed
    if mi_segment_is_in_free_queue(segment, tld) {
        mi_assert_internal((*segment).page_kind == MiPageKind::MiPageSmall); // for now we only support small pages
        mi_assert_expensive(mi_segment_queue_contains(
            addr_of_mut!((*tld).small_free),
            segment,
        ));
        mi_segment_queue_remove(addr_of_mut!((*tld).small_free), segment);
    }
    mi_assert_internal((*segment).next.is_null() && (*segment).prev.is_null());

    // all pages in the segment are abandoned; add it to the abandoned list
    (*segment).thread_id = 0;
    while !ABANDONED
        .compare_exchange(
            segment,
            (*segment).abandoned_next,
            Ordering::Relaxed,
            Ordering::SeqCst,
        )
        .is_ok()
    {
        (*segment).abandoned_next = ABANDONED.load(Ordering::Relaxed);
    }
    ABANDONED_COUNT.fetch_add(1, Ordering::Relaxed);
}

pub(crate) unsafe fn mi_segment_page_abandon(page: *mut MiPage, tld: *mut MiSegmentsTld) {
    mi_assert(!page.is_null());
    let segment = _mi_page_segment(page);
    mi_assert_expensive(mi_segment_is_valid(segment));
    (*segment.cast_mut()).abandoned += 1;
    mi_assert_internal((*segment).abandoned <= (*segment).used);
    if (*segment).used == (*segment).abandoned {
        // all pages are abandoned, abandon the entire segment
        mi_segment_abandon(segment as _, tld);
    }
}

pub(crate) unsafe fn _mi_segment_try_reclaim_abandoned(
    heap: *mut MiHeap,
    try_all: bool,
    tld: *mut MiSegmentsTld,
) -> bool {
    let mut reclaimed = 0;
    let mut atmost;
    if try_all {
        atmost = ABANDONED_COUNT.load(Ordering::Relaxed) + 16;
    } else {
        atmost = ABANDONED_COUNT.load(Ordering::Relaxed) / 8;
        if atmost < 8 {
            atmost = 8;
        }
    };

    // for `atmost` `reclaimed` abandoned segments...
    while atmost > reclaimed {
        // try to claim the head of the abandoned segments
        let mut segment = ABANDONED.load(Ordering::Relaxed);
        while !segment.is_null()
            && !ABANDONED
                .compare_exchange(
                    segment,
                    (*segment).abandoned_next,
                    Ordering::Relaxed,
                    Ordering::SeqCst,
                )
                .is_ok()
        {
            segment = ABANDONED.load(Ordering::Relaxed);
        }
        if segment.is_null() {
            break;
        } // stop early if no more segments available

        // got it.
        ABANDONED_COUNT.fetch_sub(1, Ordering::Relaxed);
        (*segment).thread_id = mi_thread_id();
        (*segment).abandoned_next = null_mut();
        mi_segments_track_size((*segment).segment_size as isize, tld);
        mi_assert_internal((*segment).next.is_null() && (*segment).prev.is_null());
        mi_assert_expensive(mi_segment_is_valid(segment));
        // add its free pages to the the current thread
        if (*segment).page_kind == MiPageKind::MiPageSmall && mi_segment_has_free(segment) {
            mi_segment_enqueue(addr_of_mut!((*tld).small_free), segment);
        }
        // add its abandoned pages to the current thread
        mi_assert((*segment).abandoned == (*segment).used);
        for i in 0..(*segment).capacity {
            let page = (*segment).pages().offset(i as _);
            if (*page).segment_in_use {
                (*segment).abandoned -= 1;
                if mi_page_all_free(page) {
                    // if everything free by now, free the page
                    mi_segment_page_clear(segment, page);
                } else {
                    // otherwise reclaim it
                    _mi_page_reclaim(heap, page);
                }
            }
        }
        mi_assert((*segment).abandoned == 0);

        if (*segment).used == 0 {
            // due to page_clear
            mi_segment_free(segment, false, tld);
        } else {
            reclaimed += 1;
        }
    }
    reclaimed > 0
}

/* -----------------------------------------------------------
   Small page allocation
----------------------------------------------------------- */

// Allocate a small page inside a (*segment).
// Requires that the page has free pages
unsafe fn mi_segment_small_page_alloc_in(
    segment: *mut MiSegment,
    tld: *mut MiSegmentsTld,
) -> *mut MiPage {
    mi_assert_internal(mi_segment_has_free(segment));
    let page = mi_segment_find_free(segment);
    (*page).segment_in_use = true;
    (*segment).used += 1;
    mi_assert_internal((*segment).used <= (*segment).capacity);
    if (*segment).used == (*segment).capacity {
        // if no more free pages, remove from the queue
        mi_assert_internal(!mi_segment_has_free(segment));
        mi_assert_expensive(mi_segment_queue_contains(
            addr_of_mut!((*tld).small_free),
            segment,
        ));
        mi_segment_queue_remove(addr_of_mut!((*tld).small_free), segment);
    }
    page
}

unsafe fn mi_segment_small_page_alloc(
    tld: *mut MiSegmentsTld,
    os_tld: *mut MiOsTld,
) -> *mut MiPage {
    if mi_segment_queue_is_empty(addr_of_mut!((*tld).small_free)) {
        let segment =
            mi_segment_alloc(0, MiPageKind::MiPageSmall, MI_SMALL_PAGE_SHIFT, tld, os_tld);
        if segment.is_null() {
            return null_mut();
        }
        mi_segment_enqueue(addr_of_mut!((*tld).small_free), segment);
    }
    mi_assert_internal((*tld).small_free.first != null_mut());
    mi_segment_small_page_alloc_in((*tld).small_free.first, tld)
}

/* -----------------------------------------------------------
   large page allocation
----------------------------------------------------------- */

unsafe fn mi_segment_large_page_alloc(
    tld: *mut MiSegmentsTld,
    os_tld: *mut MiOsTld,
) -> *mut MiPage {
    let segment = mi_segment_alloc(0, MiPageKind::MiPageLarge, MI_LARGE_PAGE_SHIFT, tld, os_tld);
    if segment.is_null() {
        return null_mut();
    }
    mi_assert_internal((*segment).segment_size - (*segment).segment_info_size >= MI_LARGE_SIZE_MAX);
    (*segment).used = 1;
    let page = (*segment).pages();
    (*page).segment_in_use = true;
    page
}

unsafe fn mi_segment_huge_page_alloc(
    size: usize,
    tld: *mut MiSegmentsTld,
    os_tld: *mut MiOsTld,
) -> *mut MiPage {
    let segment = mi_segment_alloc(size, MiPageKind::MiPageHuge, MI_SEGMENT_SHIFT, tld, os_tld);
    if segment.is_null() {
        return null_mut();
    }
    mi_assert_internal((*segment).segment_size - (*segment).segment_info_size >= size);
    (*segment).used = 1;
    let page = (*segment).pages();
    (*page).segment_in_use = true;
    page
}

/* -----------------------------------------------------------
   Page allocation and free
----------------------------------------------------------- */

pub(crate) unsafe fn _mi_segment_page_alloc(
    size: usize,
    tld: *mut MiSegmentsTld,
    os_tld: *mut MiOsTld,
) -> *mut MiPage {
    let page;
    if size < MI_SMALL_PAGE_SIZE / 8 {
        // smaller blocks than 8kb (assuming MI_SMALL_PAGE_SIZE == 64kb)
        page = mi_segment_small_page_alloc(tld, os_tld);
    } else if size < (MI_LARGE_SIZE_MAX - std::mem::size_of::<MiSegment>()) {
        page = mi_segment_large_page_alloc(tld, os_tld);
    } else {
        page = mi_segment_huge_page_alloc(size, tld, os_tld);
    }
    mi_assert_expensive(mi_segment_is_valid(_mi_page_segment(page as _)));
    page
}
