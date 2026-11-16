export interface ReturnSelection {
  libraryActivePath: string;
  multiSelectedPaths: string[];
  selectionAnchorPath: string;
}

export function getReturnSelection(
  lastPath: string | null | undefined,
  visibleList: Array<{ path: string }>,
): ReturnSelection | null {
  if (!lastPath || !visibleList.some((img) => img.path === lastPath)) return null;

  return {
    libraryActivePath: lastPath,
    multiSelectedPaths: [lastPath],
    selectionAnchorPath: lastPath,
  };
}
