import { describe, expect, it } from 'vitest';
import { LibraryViewMode } from '../components/ui/AppProperties';
import { getEditorImageList, getFolderOfPath, groupImagesByFolder, orderImagesByFolder } from './imageGrouping';

const img = (path: string) => ({ path });
const paths = (images: Array<{ path: string }>) => images.map((i) => i.path);

describe('getFolderOfPath', () => {
  it('returns the parent folder of a posix path', () => {
    expect(getFolderOfPath('/r/a/1.jpg')).toBe('/r/a');
  });

  it('returns the parent folder of a windows path', () => {
    expect(getFolderOfPath('C:\\photos\\trip\\1.jpg')).toBe('C:\\photos\\trip');
  });

  it('ignores the virtual copy suffix', () => {
    expect(getFolderOfPath('/r/a/1.jpg?vc=abc')).toBe('/r/a');
  });
});

describe('groupImagesByFolder', () => {
  it('puts the base folder first and sorts the other folders alphabetically', () => {
    const groups = groupImagesByFolder([img('/r/b/1.jpg'), img('/r/1.jpg'), img('/r/a/1.jpg')], '/r');
    expect(groups.map((g) => g.path)).toEqual(['/r', '/r/a', '/r/b']);
  });

  it('keeps the incoming order inside each folder', () => {
    const groups = groupImagesByFolder([img('/r/a/3.jpg'), img('/r/a/1.jpg'), img('/r/a/2.jpg')], '/r');
    expect(paths(groups[0].images)).toEqual(['/r/a/3.jpg', '/r/a/1.jpg', '/r/a/2.jpg']);
  });

  it('groups virtual copies with their physical folder', () => {
    const groups = groupImagesByFolder([img('/r/a/1.jpg'), img('/r/a/1.jpg?vc=1')], '/r');
    expect(groups).toHaveLength(1);
    expect(paths(groups[0].images)).toEqual(['/r/a/1.jpg', '/r/a/1.jpg?vc=1']);
  });

  it('groups windows paths', () => {
    const groups = groupImagesByFolder([img('C:\\r\\b\\1.jpg'), img('C:\\r\\a\\1.jpg'), img('C:\\r\\1.jpg')], 'C:\\r');
    expect(groups.map((g) => g.path)).toEqual(['C:\\r', 'C:\\r\\a', 'C:\\r\\b']);
  });

  it('sorts all folders alphabetically without a base folder', () => {
    expect(groupImagesByFolder([img('/b/1.jpg'), img('/a/1.jpg')], null).map((g) => g.path)).toEqual(['/a', '/b']);
    expect(groupImagesByFolder([img('/b/1.jpg'), img('/a/1.jpg')], 'Album: x').map((g) => g.path)).toEqual([
      '/a',
      '/b',
    ]);
  });
});

describe('orderImagesByFolder', () => {
  it('orders interleaved names folder by folder', () => {
    const input = [img('/r/a/1.jpg'), img('/r/b/1.jpg'), img('/r/a/2.jpg'), img('/r/2.jpg')];
    expect(paths(orderImagesByFolder(input, '/r'))).toEqual(['/r/2.jpg', '/r/a/1.jpg', '/r/a/2.jpg', '/r/b/1.jpg']);
  });

  it('returns an empty list for an empty input', () => {
    expect(orderImagesByFolder([], '/r')).toEqual([]);
  });

  it('does not mutate the input', () => {
    const input = [img('/r/b/1.jpg'), img('/r/a/1.jpg')];
    orderImagesByFolder(input, '/r');
    expect(paths(input)).toEqual(['/r/b/1.jpg', '/r/a/1.jpg']);
  });
});

describe('getEditorImageList', () => {
  const input = [img('/r/a/1.jpg'), img('/r/b/1.jpg'), img('/r/a/2.jpg')];
  const folderOrder = ['/r/a/1.jpg', '/r/a/2.jpg', '/r/b/1.jpg'];

  it('orders by folder in recursive view when sorting by folder', () => {
    const result = getEditorImageList(input, {
      libraryViewMode: LibraryViewMode.Recursive,
      sortByFolder: true,
      baseFolderPath: '/r',
    });
    expect(paths(result)).toEqual(folderOrder);
  });

  it('keeps the name order in recursive view when sorting by name', () => {
    const result = getEditorImageList(input, {
      libraryViewMode: LibraryViewMode.Recursive,
      sortByFolder: false,
      baseFolderPath: '/r',
    });
    expect(result).toBe(input);
  });

  it('keeps the list unchanged in flat view', () => {
    for (const sortByFolder of [true, false, undefined]) {
      const result = getEditorImageList(input, {
        libraryViewMode: LibraryViewMode.Flat,
        sortByFolder,
        baseFolderPath: '/r',
      });
      expect(result).toBe(input);
    }
  });

  it('defaults to folder order when the setting is unset', () => {
    const result = getEditorImageList(input, {
      libraryViewMode: LibraryViewMode.Recursive,
      sortByFolder: undefined,
      baseFolderPath: '/r',
    });
    expect(paths(result)).toEqual(folderOrder);
  });
});
