import { describe, expect, it } from 'vitest';
import { ScrollRow, centeredScrollTop, findRowOffset, resolveReveal, totalRowsHeight } from './libraryScroll';

const rows: ScrollRow[] = [
  { type: 'header' },
  { type: 'images', images: [{ path: '/r/1.jpg' }, { path: '/r/2.jpg' }] },
  { type: 'header' },
  { type: 'images', images: [{ path: '/r/a/1.jpg' }, { path: '/r/a/1.jpg?vc=1' }] },
  { type: 'footer' },
];
const sizes = [40, 200, 40, 180, 12];
const getItemSize = (i: number) => sizes[i];

describe('findRowOffset', () => {
  it('sums the heights of the rows before the match', () => {
    expect(findRowOffset(rows, getItemSize, '/r/2.jpg')).toEqual({ top: 40, height: 200 });
    expect(findRowOffset(rows, getItemSize, '/r/a/1.jpg')).toEqual({ top: 280, height: 180 });
  });

  it('returns null when the path is not in any row', () => {
    expect(findRowOffset(rows, getItemSize, '/r/b/1.jpg')).toBeNull();
    expect(findRowOffset(rows, getItemSize, null)).toBeNull();
  });

  it('matches virtual copy paths exactly', () => {
    expect(findRowOffset(rows, getItemSize, '/r/a/1.jpg?vc=1')).toEqual({ top: 280, height: 180 });
    expect(findRowOffset(rows, getItemSize, '/r/2.jpg?vc=1')).toBeNull();
  });
});

describe('centeredScrollTop', () => {
  it('centres the row in the viewport', () => {
    expect(centeredScrollTop(1000, 200, 600)).toBe(800);
  });

  it('clamps to the top of the list', () => {
    expect(centeredScrollTop(40, 200, 600)).toBe(0);
  });
});

describe('resolveReveal', () => {
  it('scrolls to a visible image', () => {
    expect(resolveReveal(rows, getItemSize, '/r/a/1.jpg', new Set())).toEqual({
      type: 'scroll',
      top: 280,
      height: 180,
    });
  });

  it('expands the collapsed folder that hides the image', () => {
    const collapsedRows: ScrollRow[] = [rows[0], rows[1], { type: 'header' }, { type: 'footer' }];
    expect(resolveReveal(collapsedRows, getItemSize, '/r/a/1.jpg?vc=1', new Set(['/r/a']))).toEqual({
      type: 'expand',
      folder: '/r/a',
    });
  });

  it('drops the request when the image is not in the list', () => {
    expect(resolveReveal(rows, getItemSize, '/other/1.jpg', new Set(['/r/a']))).toEqual({ type: 'drop' });
  });
});

describe('totalRowsHeight', () => {
  it('sums the exact height of every row', () => {
    expect(totalRowsHeight(rows.length, getItemSize)).toBe(472);
  });

  it('is large enough to reach a row near the end', () => {
    const clientHeight = 300;
    const target = findRowOffset(rows, getItemSize, '/r/a/1.jpg')!;
    const maxScrollTop = totalRowsHeight(rows.length, getItemSize) - clientHeight;
    expect(target.top + target.height).toBeLessThanOrEqual(maxScrollTop + clientHeight);
  });

  it('is 0 for an empty list', () => {
    expect(totalRowsHeight(0, getItemSize)).toBe(0);
  });
});
