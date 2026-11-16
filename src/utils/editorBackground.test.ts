import { describe, expect, it } from 'vitest';
import { NEUTRAL_GREY_RGB, hexToRgba01, parseRgb, resolveEditorBackground, rgbaToHex } from './editorBackground';

const THEME: [number, number, number, number] = [35 / 255, 35 / 255, 35 / 255, 1];

describe('hexToRgba01', () => {
  it('converts a 6 digit hex', () => {
    expect(hexToRgba01('#808080')).toEqual([128 / 255, 128 / 255, 128 / 255, 1]);
  });
  it('expands shorthand', () => {
    expect(hexToRgba01('#fff')).toEqual([1, 1, 1, 1]);
  });
  it('is case-insensitive', () => {
    expect(hexToRgba01('#FF0000')).toEqual([1, 0, 0, 1]);
  });
  it('returns null for invalid input', () => {
    expect(hexToRgba01('red')).toBeNull();
    expect(hexToRgba01('#12')).toBeNull();
    expect(hexToRgba01('')).toBeNull();
    expect(hexToRgba01(null)).toBeNull();
  });
});

describe('resolveEditorBackground', () => {
  it('custom colour overrides neutral grey', () => {
    const r = resolveEditorBackground({ customColor: '#ff0000', neutralGrey: true, themeRgb: THEME });
    expect(r.rgba).toEqual([1, 0, 0, 1]);
    expect(r.css).toBe('rgb(255, 0, 0)');
  });
  it('uses neutral grey without a custom colour', () => {
    const r = resolveEditorBackground({ customColor: null, neutralGrey: true, themeRgb: THEME });
    expect(r.rgba).toEqual(NEUTRAL_GREY_RGB);
    expect(r.css).toBe('#808080');
  });
  it('uses the theme when nothing is set', () => {
    const r = resolveEditorBackground({ themeRgb: THEME });
    expect(r.rgba).toEqual(THEME);
    expect(r.css).toBeNull();
  });
  it('falls back to the theme for an invalid custom colour', () => {
    const r = resolveEditorBackground({ customColor: 'nope', themeRgb: THEME });
    expect(r.rgba).toEqual(THEME);
    expect(r.css).toBeNull();
  });
});

describe('parseRgb', () => {
  it('parses rgb()', () => {
    expect(parseRgb('rgb(35, 35, 35)')).toEqual(THEME);
  });
  it('falls back to black', () => {
    expect(parseRgb('')).toEqual([0, 0, 0, 1]);
  });
});

describe('rgbaToHex', () => {
  it('converts back to hex', () => {
    expect(rgbaToHex(NEUTRAL_GREY_RGB)).toBe('#808080');
    expect(rgbaToHex(parseRgb('rgb(118, 118, 118)'))).toBe('#767676');
  });
  it('round-trips with hexToRgba01', () => {
    expect(rgbaToHex(hexToRgba01('#1a2b3c')!)).toBe('#1a2b3c');
  });
});
