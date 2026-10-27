import { useEffect, useMemo, useRef, useState } from 'react';
import { HexColorInput, HexColorPicker } from 'react-colorful';
import { useTranslation } from 'react-i18next';
import clsx from 'clsx';
import { useSettingsStore } from '../store/useSettingsStore';
import { parseRgb, resolveEditorBackground, rgbaToHex } from '../utils/editorBackground';

const PRESETS = [
  { color: '#000000', label: 'contextMenus.editor.bgBlack' },
  { color: '#232323', label: 'contextMenus.editor.bgDarkGrey' },
  { color: '#808080', label: 'contextMenus.editor.bgNeutralGrey' },
  { color: '#c0c0c0', label: 'contextMenus.editor.bgLightGrey' },
  { color: '#ffffff', label: 'contextMenus.editor.bgWhite' },
] as const;

const SAVE_DEBOUNCE_MS = 300;

export default function BackgroundColorSubMenu() {
  const { t } = useTranslation();
  const stored = useSettingsStore((s) => s.appSettings?.editorBackgroundColor ?? null);
  const neutralGrey = useSettingsStore((s) => s.appSettings?.editorNeutralGreyBg ?? false);
  const theme = useSettingsStore((s) => s.appSettings?.theme);
  const [color, setColor] = useState<string | null>(stored);

  // With no custom colour, show what the canvas actually uses (neutral grey or the theme background).
  const fallbackColor = useMemo(() => {
    const themeStr = getComputedStyle(document.documentElement).getPropertyValue('--app-bg-secondary');
    return rgbaToHex(resolveEditorBackground({ neutralGrey, themeRgb: parseRgb(themeStr) }).rgba);
  }, [neutralGrey, theme]);
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pending = useRef<{ value: string | null } | null>(null);

  const persist = () => {
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = null;
    if (!pending.current) return;
    const { appSettings, handleSettingsChange } = useSettingsStore.getState();
    const value = pending.current.value;
    pending.current = null;
    if (appSettings) handleSettingsChange({ ...appSettings, editorBackgroundColor: value });
  };

  useEffect(() => persist, []);

  const apply = (value: string | null, immediate: boolean) => {
    setColor(value);
    const { appSettings, setAppSettings } = useSettingsStore.getState();
    if (!appSettings) return;
    setAppSettings({ ...appSettings, editorBackgroundColor: value });
    pending.current = { value };
    if (immediate) {
      persist();
    } else {
      if (saveTimer.current) clearTimeout(saveTimer.current);
      saveTimer.current = setTimeout(persist, SAVE_DEBOUNCE_MS);
    }
  };

  const swatch = 'w-6 h-6 rounded-md border border-border-color transition-transform hover:scale-110';
  const ring = 'ring-2 ring-accent';

  return (
    <div
      className="bg-surface/95 p-3 rounded-lg w-56 flex flex-col gap-3"
      onPointerDown={(e) => e.stopPropagation()}
      onKeyDown={(e) => e.stopPropagation()}
    >
      <div className="flex items-center gap-2">
        <button
          type="button"
          className={clsx(
            swatch,
            'bg-bg-secondary text-[10px] text-text-secondary flex items-center justify-center',
            color === null && ring,
          )}
          data-tooltip={t('contextMenus.editor.themeDefault')}
          title={t('contextMenus.editor.themeDefault')}
          onClick={() => apply(null, true)}
        >
          T
        </button>
        {PRESETS.map((p) => (
          <button
            key={p.color}
            type="button"
            className={clsx(swatch, color?.toLowerCase() === p.color && ring)}
            style={{ backgroundColor: p.color }}
            title={t(p.label)}
            onClick={() => apply(p.color, true)}
          />
        ))}
      </div>
      <HexColorPicker
        color={color ?? fallbackColor}
        onChange={(c) => apply(c, false)}
        style={{ width: '100%', height: 140 }}
      />
      <HexColorInput
        prefixed
        color={color ?? fallbackColor}
        onChange={(c) => apply(c, false)}
        className="w-full bg-bg-primary border border-border-color rounded-md px-2 py-1 text-sm text-text-primary outline-none focus:border-accent"
      />
    </div>
  );
}
