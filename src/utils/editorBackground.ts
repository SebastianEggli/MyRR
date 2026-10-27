export type Rgba = [number, number, number, number];

export const NEUTRAL_GREY_RGB: Rgba = [128 / 255, 128 / 255, 128 / 255, 1.0];

export const parseRgb = (rgbStr: string): Rgba => {
  const match = rgbStr.match(/[\d.]+/g);
  if (match && match.length >= 3) {
    return [parseFloat(match[0]) / 255, parseFloat(match[1]) / 255, parseFloat(match[2]) / 255, 1.0];
  }
  return [0, 0, 0, 1.0];
};

export const hexToRgba01 = (hex: string | null | undefined): Rgba | null => {
  if (!hex) return null;
  const m = hex.trim().match(/^#?([0-9a-f]{3}|[0-9a-f]{6})$/i);
  if (!m) return null;
  let digits = m[1];
  if (digits.length === 3) {
    digits = digits
      .split('')
      .map((c) => c + c)
      .join('');
  }
  const n = parseInt(digits, 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255, 1.0];
};

export const rgbaToHex = ([r, g, b]: Rgba): string =>
  '#' +
  [r, g, b]
    .map((v) =>
      Math.round(Math.min(1, Math.max(0, v)) * 255)
        .toString(16)
        .padStart(2, '0'),
    )
    .join('');

interface ResolveArgs {
  customColor?: string | null;
  neutralGrey?: boolean;
  themeRgb: Rgba;
}

export const resolveEditorBackground = ({
  customColor,
  neutralGrey,
  themeRgb,
}: ResolveArgs): { rgba: Rgba; css: string | null } => {
  const custom = hexToRgba01(customColor);
  if (custom) {
    const [r, g, b] = custom.map((v) => Math.round(v * 255));
    return { rgba: custom, css: `rgb(${r}, ${g}, ${b})` };
  }
  if (neutralGrey) {
    return { rgba: NEUTRAL_GREY_RGB, css: '#808080' };
  }
  return { rgba: themeRgb, css: null };
};
