"""Four discrete mask opacities, shared safely by Tk and the render worker."""
import threading

MASK_LEVELS = ('low', 'midium', 'strong', 'very strong')
MASK_STRENGTHS = (.25, .50, .75, 1.)


class MaskStrengthControl:
    def __init__(self, level=3):
        self._lock = threading.Lock()
        self.set_level(level)

    def set_level(self, level):
        if isinstance(level, bool) or not isinstance(level, int) or not 0 <= level < len(MASK_LEVELS):
            raise ValueError('마스킹 강도는 0~3 단계여야 합니다.')
        with self._lock:
            self._level = level

    def snapshot(self):
        with self._lock:
            level = self._level
        return MASK_LEVELS[level], MASK_STRENGTHS[level]
