from __future__ import annotations

import queue
import threading
import tkinter as tk
import uuid
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .faces import Landmarker, composite, extract_face
from .media import Cancelled, ROOT, WORKSPACE, normalize_video, read_frame
from .pipeline import render_video
from .tracking import SelectedTracker
from .identity import CompositeSettings


class FaceSwapGUI:
    WIDTH, HEIGHT = 832, 468

    def __init__(self, root):
        self.root = root
        root.title('FaceSwab2 · 선택한 얼굴 합성')
        root.geometry('1160x900')
        root.minsize(1120, 860)
        self.events = queue.Queue()
        self.cancel = threading.Event()
        self.busy = False
        self.closing = False
        self.video = self.original = self.asset = self.frame = self.info = None
        self.index = 0
        self.selections = {}
        self.drag_start = None
        self.rect = None
        self.controls = []
        self.force_region = tk.BooleanVar(value=False)
        self.preset_name = tk.StringVar(value='균형')
        self.setting_vars = {name: tk.DoubleVar(value=value) for name, value in
                             vars(CompositeSettings.preset('balanced')).items()}
        self.status = tk.StringVar(value='영상을 가져온 뒤, 바꿀 얼굴이 보이는 시점에서 드래그하세요.')
        self.position = tk.StringVar(value='0.00초 / 0.00초')
        self.video_label = tk.StringVar(value='선택된 영상 없음')
        self.face_label = tk.StringVar(value='JPEG · PNG 정면 얼굴 한 명')
        self.selection_label = tk.StringVar(value='선택 시점 없음')
        self._build()
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(80, self.poll)

    def button(self, parent, text, command):
        button = ttk.Button(parent, text=text, command=command)
        self.controls.append(button)
        return button

    def _build(self):
        shell = ttk.Frame(self.root, padding=16)
        shell.pack(fill='both', expand=True)
        ttk.Label(shell, text='선택한 캐릭터에 내 얼굴 입히기', font=('맑은 고딕', 19, 'bold')).pack(anchor='w')
        ttk.Label(shell, text='얼굴 이미지 추출 → 영상에서 얼굴 드래그 → 미리보기 → 720p MP4 저장').pack(anchor='w', pady=(4, 14))
        content = ttk.Frame(shell)
        content.pack(fill='both', expand=True)
        left = ttk.Frame(content)
        left.pack(side='left', anchor='n')
        toolbar = ttk.Frame(left)
        toolbar.pack(fill='x', pady=(0, 8))
        self.button(toolbar, '1. 영상 가져오기', self.load_video).pack(side='left')
        ttk.Label(toolbar, textvariable=self.video_label, width=70).pack(side='left', padx=10)
        self.canvas = tk.Canvas(left, width=self.WIDTH, height=self.HEIGHT, bg='#171d29', highlightthickness=0)
        self.canvas.pack()
        self.canvas.create_text(self.WIDTH/2, self.HEIGHT/2, text='영상을 가져오면 여기에 표시됩니다', fill='#b8c4d9', font=('맑은 고딕', 14))
        self.canvas.bind('<ButtonPress-1>', self.drag_begin)
        self.canvas.bind('<B1-Motion>', self.drag_move)
        self.canvas.bind('<ButtonRelease-1>', self.drag_end)
        self.slider = ttk.Scale(left, from_=0, to=1, orient='horizontal')
        self.slider.pack(fill='x', pady=(8, 0))
        self.slider.bind('<ButtonRelease-1>', self.seek)
        self.slider.bind('<KeyRelease>', self.seek)
        self.controls.append(self.slider)
        row = ttk.Frame(left)
        row.pack(fill='x')
        self.button(row, '◀ 이전 프레임', lambda: self.step(-1)).pack(side='left')
        self.button(row, '다음 프레임 ▶', lambda: self.step(1)).pack(side='left', padx=5)
        ttk.Label(row, textvariable=self.position).pack(side='right')
        ttk.Label(left, text='선택한 시점부터 합성합니다. 장면이 바뀌거나 추적을 잃으면 새 시점에서 얼굴을 다시 선택하세요.', wraplength=825).pack(anchor='w', pady=(8, 0))
        right = ttk.Frame(content, padding=(16, 0, 0, 0))
        right.pack(side='left', fill='both', expand=True)
        self.button(right, '2. 얼굴 이미지 가져오기', self.load_face).pack(fill='x')
        ttk.Label(right, textvariable=self.face_label, wraplength=220).pack(anchor='w', pady=8)
        self.face_canvas = tk.Canvas(right, width=220, height=220, bg='#cdd2da', highlightthickness=0)
        self.face_canvas.pack()
        self.button(right, '투명 얼굴 PNG 저장', self.save_face).pack(fill='x', pady=(8, 18))
        ttk.Label(right, text='3. 영상의 얼굴을 드래그', font=('맑은 고딕', 11, 'bold')).pack(anchor='w')
        ttk.Label(right, textvariable=self.selection_label, wraplength=220).pack(anchor='w', pady=8)
        check = ttk.Checkbutton(right, text='영역 모드 (표정 변형 없음)', variable=self.force_region, command=self.region_changed)
        check.pack(anchor='w')
        self.controls.append(check)
        ttk.Label(right, text='정밀 모드: 눈꺼풀·입술·볼 움직임 반영\n얼굴 인식 실패 시 영역 추적으로 전환', wraplength=220).pack(anchor='w', pady=6)
        self.button(right, '선택 시점 미리보기', self.refresh_preview).pack(fill='x', pady=3)
        self.button(right, '현재 시점 선택 지우기', self.clear_selection).pack(fill='x', pady=3)
        self.button(right, '모든 선택 지우기', self.clear_all).pack(fill='x', pady=3)
        tuning = ttk.LabelFrame(left, text='교체 얼굴 특징 · 값을 바꾼 뒤 선택 시점 미리보기', padding=6)
        tuning.pack(fill='x', pady=(8, 0))
        self.preset_combo = ttk.Combobox(tuning, textvariable=self.preset_name,
                                        values=('기존', '균형', '특징 강화'), state='readonly', width=10)
        self.preset_combo.grid(row=0, column=0, padx=4)
        self.preset_combo.bind('<<ComboboxSelected>>', self.apply_preset)
        self.controls.append(self.preset_combo)
        self.setting_controls = {}
        for column, (name, label) in enumerate((('identity', '얼굴 비율'), ('lighting', '밝기 보정'),
                                                ('skin_color', '피부색 보정'), ('detail', '원본 명암')), 1):
            cell = ttk.Frame(tuning)
            cell.grid(row=0, column=column, padx=9)
            ttk.Label(cell, text=label).pack()
            spin = ttk.Spinbox(cell, from_=0, to=.65 if name == 'identity' else 1., increment=.05,
                               textvariable=self.setting_vars[name], width=7, format='%.2f',
                               command=lambda: self.preset_name.set('사용자 설정'))
            spin.bind('<KeyRelease>', lambda event: self.preset_name.set('사용자 설정'))
            spin.pack()
            self.setting_controls[name] = spin
            self.controls.append(spin)
        bottom = ttk.Frame(shell)
        bottom.pack(fill='x', pady=(12, 0))
        self.button(bottom, '4. 합성 영상 저장', self.export).pack(side='left')
        self.cancel_button = ttk.Button(bottom, text='취소', command=self.cancel.set, state='disabled')
        self.cancel_button.pack(side='left', padx=8)
        ttk.Label(bottom, text='1280×720 · MP4/H.264 · AAC 스테레오 · 30fps').pack(side='right')
        self.progress = ttk.Progressbar(shell, maximum=100)
        self.progress.pack(fill='x', pady=8)
        ttk.Label(shell, textvariable=self.status, wraplength=1090).pack(anchor='w')

    def set_busy(self, value):
        self.busy = value
        for control in self.controls:
            control.configure(state='disabled' if value else 'normal')
        self.cancel_button.configure(state='normal' if value else 'disabled')
        if not value:
            self.preset_combo.configure(state='readonly')
            self.setting_controls['identity'].configure(state='disabled' if self.force_region.get() else 'normal')

    def composite_settings(self):
        return CompositeSettings(**{name: value.get() for name, value in self.setting_vars.items()})

    def apply_preset(self, event=None):
        name = {'기존': 'legacy', '균형': 'balanced', '특징 강화': 'strong'}[self.preset_name.get()]
        for key, value in vars(CompositeSettings.preset(name)).items():
            self.setting_vars[key].set(value)
        self.refresh_preview()

    def region_changed(self):
        self.setting_controls['identity'].configure(state='disabled' if self.force_region.get() else 'normal')
        self.refresh_preview()

    def submit(self, work, done, label):
        if self.busy:
            return
        self.cancel.clear()
        self.set_busy(True)
        self.progress['value'] = 0
        self.status.set(label)
        def run():
            try:
                self.events.put(('done', done, work()))
            except Exception as error:
                self.events.put(('error', error))
        threading.Thread(target=run, daemon=True).start()

    def poll(self):
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == 'progress':
                    self.progress['value'] = event[1]*100
                    self.status.set(event[2])
                elif event[0] == 'done':
                    self.set_busy(False)
                    if not self.closing:
                        event[1](event[2])
                elif event[0] == 'error':
                    self.set_busy(False)
                    self.status.set(str(event[1]))
                    if not self.closing and not isinstance(event[1], Cancelled):
                        messagebox.showerror('처리 실패', str(event[1]), parent=self.root)
        except queue.Empty:
            pass
        if self.closing and not self.busy:
            self.root.destroy()
        else:
            self.root.after(80, self.poll)

    def load_video(self):
        filename = filedialog.askopenfilename(title='합성할 영상', filetypes=[('영상', '*.mp4 *.mov *.mkv *.avi *.webm'), ('모든 파일', '*.*')])
        if not filename:
            return
        destination = WORKSPACE / f'{uuid.uuid4().hex}.mp4'
        def work():
            try:
                info = normalize_video(filename, destination, self.cancel)
                return info, read_frame(destination)
            except Exception:
                destination.unlink(missing_ok=True)
                raise
        def done(result):
            self.info, self.frame = result
            self.video, self.original = destination, Path(filename).resolve()
            self.selections.clear()
            self.index = 0
            self.slider.configure(to=self.info['frames']-1)
            self.slider.set(0)
            self.video_label.set(Path(filename).name)
            self.selection_summary()
            self.show_frame(self.frame)
            self.update_position()
            self.progress['value'] = 100
            self.status.set('영상 준비 완료. 슬라이더로 시점을 고른 뒤, 얼굴 윤곽을 감싸도록 드래그하세요.')
        self.submit(work, done, '영상 가져오는 중 · 720p MP4로 변환합니다…')

    def load_face(self):
        filename = filedialog.askopenfilename(title='정면 얼굴 이미지', filetypes=[('얼굴 이미지', '*.jpg *.jpeg *.png')])
        if not filename:
            return
        destination = WORKSPACE / f'face-{uuid.uuid4().hex}.png'
        def work():
            model = Landmarker()
            try:
                return extract_face(filename, destination, model)
            finally:
                model.close()
        def done(asset):
            self.asset = asset
            self.face_label.set(Path(filename).name + '\n얼굴 윤곽 추출 · 헤어 분리 완료')
            self.show_cutout()
            self.status.set('배경·머리카락·목을 제외한 얼굴을 추출했습니다. 영상 속 얼굴을 선택하세요.')
            if self.frame is not None:
                self.show_frame(self.frame)
        self.submit(work, done, '정면 얼굴을 인식하고 투명 배경으로 추출하는 중…')

    def show_cutout(self):
        size = 220
        grid = ((np.arange(size)[:, None]//12 + np.arange(size)[None, :]//12) % 2)
        background = Image.fromarray(np.repeat(np.where(grid[..., None], 224, 194).astype(np.uint8), 3, axis=2)).convert('RGBA')
        with Image.open(self.asset.path) as opened:
            face = opened.convert('RGBA')
        face.thumbnail((210, 210))
        background.alpha_composite(face, ((size-face.width)//2, (size-face.height)//2))
        self.face_photo = ImageTk.PhotoImage(background)
        self.face_canvas.delete('all')
        self.face_canvas.create_image(0, 0, image=self.face_photo, anchor='nw')

    def save_face(self):
        if not self.asset:
            messagebox.showinfo('얼굴 이미지', '먼저 얼굴 이미지를 가져오세요.')
            return
        path = filedialog.asksaveasfilename(title='투명 얼굴 저장', defaultextension='.png', filetypes=[('PNG', '*.png')])
        if path:
            try:
                Path(path).write_bytes(self.asset.path.read_bytes())
                self.status.set(f'투명 얼굴 저장: {path}')
            except OSError as error:
                messagebox.showerror('저장 실패', str(error))

    def show_frame(self, frame, preview=False):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.video_photo = ImageTk.PhotoImage(Image.fromarray(rgb).resize((self.WIDTH, self.HEIGHT)))
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, image=self.video_photo, anchor='nw')
        self.rect = None
        if self.index in self.selections:
            x, y, w, h = self.selections[self.index]
            scale = self.WIDTH/1280
            self.rect = self.canvas.create_rectangle(x*scale, y*scale, (x+w)*scale, (y+h)*scale, outline='#4ade80', width=2)
        if preview:
            self.canvas.create_text(12, 12, text='선택 시점 합성 미리보기', fill='#4ade80', anchor='nw')

    def update_position(self):
        self.position.set(f'{self.index/self.info["fps"]:.2f}초 / {self.info["duration"]:.2f}초 · {self.index+1} 프레임')

    def seek(self, event=None):
        if not self.video or self.busy:
            return
        try:
            index = min(self.info['frames']-1, max(0, round(self.slider.get())))
            frame = read_frame(self.video, index)
            self.index, self.frame = index, frame
            self.show_frame(frame)
            self.update_position()
        except Exception as error:
            self.status.set(str(error))

    def step(self, amount):
        if self.video and not self.busy:
            self.slider.set(min(self.info['frames']-1, max(0, self.index+amount)))
            self.seek()

    def coordinates(self, event):
        return min(self.WIDTH, max(0, event.x)), min(self.HEIGHT, max(0, event.y))

    def drag_begin(self, event):
        if self.frame is None or self.busy:
            return
        self.show_frame(self.frame)
        self.drag_start = self.coordinates(event)
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(*self.drag_start, *self.drag_start, outline='#4ade80', width=2)

    def drag_move(self, event):
        if self.drag_start is not None and not self.busy:
            self.canvas.coords(self.rect, *self.drag_start, *self.coordinates(event))

    def drag_end(self, event):
        if self.drag_start is None or self.busy:
            return
        x0, y0 = self.drag_start
        x1, y1 = self.coordinates(event)
        self.drag_start = None
        scale = 1280/self.WIDTH
        box = (round(min(x0, x1)*scale), round(min(y0, y1)*scale), round(abs(x1-x0)*scale), round(abs(y1-y0)*scale))
        if min(box[2:]) < 24:
            self.status.set('얼굴 전체를 포함하도록 조금 더 크게 드래그하세요.')
            self.show_frame(self.frame)
            return
        self.selections[self.index] = box
        self.selection_summary()
        self.show_frame(self.frame)
        self.refresh_preview()

    def selection_summary(self):
        if not self.selections:
            self.selection_label.set('선택 시점 없음')
        else:
            seconds = [f'{i/self.info["fps"]:.2f}초' for i in sorted(self.selections)]
            self.selection_label.set(f'{len(seconds)}개 선택 시점\n' + ', '.join(seconds[-6:]))

    def clear_selection(self):
        self.selections.pop(self.index, None)
        self.selection_summary()
        if self.frame is not None:
            self.show_frame(self.frame)

    def clear_all(self):
        self.selections.clear()
        self.clear_selection()

    def refresh_preview(self):
        if self.busy or self.frame is None:
            return
        if self.index not in self.selections:
            self.status.set('현재 시점에서 얼굴을 드래그하면 합성 미리보기를 볼 수 있습니다.')
            return
        if self.asset is None:
            self.status.set('선택 완료. 얼굴 이미지를 가져온 뒤 미리보기를 눌러주세요.')
            return
        frame, box, asset, force = self.frame.copy(), self.selections[self.index], self.asset, self.force_region.get()
        try:
            settings = self.composite_settings()
        except (ValueError, tk.TclError) as error:
            messagebox.showerror('합성 설정', str(error), parent=self.root)
            return
        def work():
            model = Landmarker()
            try:
                tracker = SelectedTracker(frame, box, model, force)
                return composite(frame, asset, tracker.points, tracker.box, landmarker=model, settings=settings), tracker.mode
            finally:
                model.close()
        def done(result):
            frame, mode = result
            self.show_frame(frame, True)
            self.status.set('정밀 표정 모드 · 눈꺼풀·입술·볼 변형 / 영상의 눈동자·입안 보존 / 헤어 제외' if mode == 'mesh' else
                            '영역 추적 모드 · 이 얼굴은 정밀 표정을 인식하지 못합니다. 표정·회전 변형은 적용되지 않습니다.')
        self.submit(work, done, '선택한 얼굴의 합성 미리보기 만드는 중…')

    def export(self):
        if not self.video or not self.asset or not self.selections:
            messagebox.showinfo('입력 확인', '영상과 얼굴 이미지를 가져온 뒤 영상 속 얼굴을 드래그하세요.')
            return
        try:
            settings = self.composite_settings()
        except (ValueError, tk.TclError) as error:
            messagebox.showerror('합성 설정', str(error), parent=self.root)
            return
        filename = filedialog.asksaveasfilename(title='합성 MP4 저장', initialdir=str(ROOT/'outputs'),
                                               initialfile='face_result.mp4', defaultextension='.mp4', filetypes=[('MP4', '*.mp4')])
        if not filename:
            return
        destination = Path(filename).resolve()
        if destination in {self.original, self.video.resolve()}:
            messagebox.showerror('출력 경로', '입력 영상과 다른 출력 파일을 선택하세요.')
            return
        video, asset, selections, force = self.video, self.asset, dict(self.selections), self.force_region.get()
        def work():
            model = Landmarker()
            try:
                return render_video(video, asset, selections, destination, model,
                                    lambda p, s: self.events.put(('progress', p, s)), self.cancel, force, settings=settings)
            finally:
                model.close()
        def done(report):
            skipped = sum(b-a+1 for a, b in report['skipped_tracking_ranges'])
            self.status.set(f'저장 완료: {destination} · 합성 {report["replaced_frames"]}프레임 · 추적 중단 {skipped}프레임')
            detail = f'720p MP4 · 스테레오 저장 완료\n{destination}\n\n합성: {report["replaced_frames"]}/{report["frames"]}프레임'
            modes = report['mode_frames']
            detail += f'\n정밀 표정: {modes["mesh"]}프레임 · 영역 추적: {modes["region"]}프레임'
            if modes['region']:
                detail += '\n영역 추적 구간에는 표정 변형이 적용되지 않았습니다.'
            if skipped:
                detail += f'\n추적을 잃은 {skipped}프레임은 원본입니다. 해당 구간에 선택 시점을 추가하여 다시 합성하세요.'
            messagebox.showinfo('합성 완료', detail)
        self.submit(work, done, '선택한 얼굴 추적 및 합성을 시작합니다…')

    def close(self):
        if self.busy:
            self.closing = True
            self.cancel.set()
            self.status.set('작업을 정리한 뒤 종료합니다…')
        else:
            self.root.destroy()


def main():
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    (ROOT/'outputs').mkdir(exist_ok=True)
    root = tk.Tk()
    ttk.Style(root).theme_use('clam')
    FaceSwapGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
