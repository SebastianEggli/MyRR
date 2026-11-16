import { getFolderOfPath } from './imageGrouping';

export interface ScrollRow {
  type: string;
  images?: Array<{ path: string }>;
}

export function findRowOffset(
  rows: ScrollRow[],
  getItemSize: (index: number) => number,
  path: string | null,
): { top: number; height: number } | null {
  if (!path) return null;

  let top = 0;
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    const height = getItemSize(i);

    if (row.type === 'images' && row.images?.some((img) => img.path === path)) {
      return { top, height };
    }

    top += height;
  }

  return null;
}

export type RevealAction =
  { type: 'scroll'; top: number; height: number } | { type: 'expand'; folder: string } | { type: 'drop' };

export function resolveReveal(
  rows: ScrollRow[],
  getItemSize: (index: number) => number,
  path: string,
  collapsedFolders: Set<string>,
): RevealAction {
  const target = findRowOffset(rows, getItemSize, path);
  if (target) return { type: 'scroll', ...target };

  const folder = getFolderOfPath(path);
  if (collapsedFolders.has(folder)) return { type: 'expand', folder };

  return { type: 'drop' };
}

export function totalRowsHeight(rowCount: number, getItemSize: (index: number) => number): number {
  let total = 0;
  for (let i = 0; i < rowCount; i++) {
    total += getItemSize(i);
  }
  return total;
}

export function centeredScrollTop(top: number, rowHeight: number, clientHeight: number): number {
  return Math.max(0, top - clientHeight / 2 + rowHeight / 2);
}
