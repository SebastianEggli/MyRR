import { describe, expect, it } from 'vitest';
import { getReturnSelection } from './librarySelection';

const visible = [{ path: '/r/1.jpg' }, { path: '/r/a/2.jpg' }];

describe('getReturnSelection', () => {
  it('selects only the last edited image when it is still visible', () => {
    expect(getReturnSelection('/r/a/2.jpg', visible)).toEqual({
      libraryActivePath: '/r/a/2.jpg',
      multiSelectedPaths: ['/r/a/2.jpg'],
      selectionAnchorPath: '/r/a/2.jpg',
    });
  });

  it('returns null when the image is no longer visible', () => {
    expect(getReturnSelection('/r/deleted.jpg', visible)).toBeNull();
  });

  it('returns null without a last edited image', () => {
    expect(getReturnSelection(null, visible)).toBeNull();
    expect(getReturnSelection(undefined, visible)).toBeNull();
  });
});
