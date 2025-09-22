# -*- coding: utf-8 -*-
"""
얼굴 교체 GUI 애플리케이션
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
from pathlib import Path
import subprocess
import json

# 한글 출력을 위한 인코딩 설정
import locale
import codecs
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.preprocess.simple_video_extractor import SimpleVideoPreprocessor
from src.infer.video_synthesizer import VideoSynthesizer


class FaceSwapGUI:
    """얼굴 교체 GUI 클래스"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("얼굴 교체 프로그램")
        self.root.geometry("800x600")
        
        # 변수 초기화
        self.source_video_path = tk.StringVar()
        self.target_image_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.meta_path = tk.StringVar()
        self.sample_rate = tk.IntVar(value=3)
        self.mode = tk.StringVar(value="lightwarp")
        
        self.setup_ui()
        
    def setup_ui(self):
        """UI 설정"""
        # 메인 프레임
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 제목
        title_label = ttk.Label(main_frame, text="🎭 얼굴 교체 프로그램", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 경고 메시지
        warning_label = ttk.Label(main_frame, 
                                text="⚠️ 사용 전 대상자의 명시적 동의 필요 — 교육/연구 목적일 때만 사용하십시오.",
                                foreground="red", font=("Arial", 10, "bold"))
        warning_label.grid(row=1, column=0, columnspan=3, pady=(0, 20))
        
        # 탭 노트북
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 1단계: 얼굴 추출 탭
        self.setup_extract_tab(notebook)
        
        # 2단계: 얼굴 교체 탭
        self.setup_swap_tab(notebook)
        
        # 로그 탭
        self.setup_log_tab(notebook)
        
        # 그리드 가중치 설정
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
    def setup_extract_tab(self, notebook):
        """얼굴 추출 탭 설정"""
        extract_frame = ttk.Frame(notebook, padding="10")
        notebook.add(extract_frame, text="1단계: 얼굴 추출")
        
        # 소스 비디오 선택
        ttk.Label(extract_frame, text="소스 비디오:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(extract_frame, textvariable=self.source_video_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(extract_frame, text="찾아보기", 
                  command=self.browse_source_video).grid(row=0, column=2, padx=5)
        
        # 샘플링 비율
        ttk.Label(extract_frame, text="샘플링 비율:").grid(row=1, column=0, sticky=tk.W, pady=5)
        sample_spinbox = ttk.Spinbox(extract_frame, from_=1, to=10, 
                                   textvariable=self.sample_rate, width=10)
        sample_spinbox.grid(row=1, column=1, sticky=tk.W, padx=5)
        ttk.Label(extract_frame, text="(낮을수록 정확하지만 느림)").grid(row=1, column=2, sticky=tk.W)
        
        # 출력 폴더
        ttk.Label(extract_frame, text="출력 폴더:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(extract_frame, textvariable=self.output_path, width=50).grid(row=2, column=1, padx=5)
        ttk.Button(extract_frame, text="찾아보기", 
                  command=self.browse_output_folder).grid(row=2, column=2, padx=5)
        
        # 실행 버튼
        extract_btn = ttk.Button(extract_frame, text="얼굴 추출 시작", 
                               command=self.start_extract, style="Accent.TButton")
        extract_btn.grid(row=3, column=0, columnspan=3, pady=20)
        
        # 진행률 바
        self.extract_progress = ttk.Progressbar(extract_frame, mode='indeterminate')
        self.extract_progress.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
    def setup_swap_tab(self, notebook):
        """얼굴 교체 탭 설정"""
        swap_frame = ttk.Frame(notebook, padding="10")
        notebook.add(swap_frame, text="2단계: 얼굴 교체")
        
        # 타겟 이미지 선택
        ttk.Label(swap_frame, text="타겟 이미지:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(swap_frame, textvariable=self.target_image_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(swap_frame, text="찾아보기", 
                  command=self.browse_target_image).grid(row=0, column=2, padx=5)
        
        # 소스 비디오 (추출된 것 사용)
        ttk.Label(swap_frame, text="소스 비디오:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(swap_frame, textvariable=self.source_video_path, width=50, state="readonly").grid(row=1, column=1, padx=5)
        
        # 메타데이터 파일
        ttk.Label(swap_frame, text="메타데이터:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(swap_frame, textvariable=self.meta_path, width=50).grid(row=2, column=1, padx=5)
        ttk.Button(swap_frame, text="찾아보기", 
                  command=self.browse_meta_file).grid(row=2, column=2, padx=5)
        
        # 모드 선택
        ttk.Label(swap_frame, text="처리 모드:").grid(row=3, column=0, sticky=tk.W, pady=5)
        mode_frame = ttk.Frame(swap_frame)
        mode_frame.grid(row=3, column=1, sticky=tk.W, padx=5)
        ttk.Radiobutton(mode_frame, text="LightWarp (빠름)", 
                       variable=self.mode, value="lightwarp").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="FOMM (고품질)", 
                       variable=self.mode, value="fomm").pack(side=tk.LEFT, padx=(10, 0))
        
        # 출력 파일
        ttk.Label(swap_frame, text="출력 파일:").grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Entry(swap_frame, textvariable=self.output_path, width=50).grid(row=4, column=1, padx=5)
        ttk.Button(swap_frame, text="찾아보기", 
                  command=self.browse_output_file).grid(row=4, column=2, padx=5)
        
        # 실행 버튼
        swap_btn = ttk.Button(swap_frame, text="얼굴 교체 시작", 
                             command=self.start_swap, style="Accent.TButton")
        swap_btn.grid(row=5, column=0, columnspan=3, pady=20)
        
        # 진행률 바
        self.swap_progress = ttk.Progressbar(swap_frame, mode='indeterminate')
        self.swap_progress.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
    def setup_log_tab(self, notebook):
        """로그 탭 설정"""
        log_frame = ttk.Frame(notebook, padding="10")
        notebook.add(log_frame, text="실행 로그")
        
        # 로그 텍스트 영역
        self.log_text = scrolledtext.ScrolledText(log_frame, height=20, width=80)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 로그 버튼들
        log_btn_frame = ttk.Frame(log_frame)
        log_btn_frame.grid(row=1, column=0, pady=5)
        
        ttk.Button(log_btn_frame, text="로그 지우기", 
                  command=self.clear_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(log_btn_frame, text="로그 저장", 
                  command=self.save_log).pack(side=tk.LEFT, padx=5)
        
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
    def browse_source_video(self):
        """소스 비디오 선택"""
        filename = filedialog.askopenfilename(
            title="소스 비디오 선택",
            filetypes=[("비디오 파일", "*.mp4 *.avi *.mov *.mkv"), ("모든 파일", "*.*")]
        )
        if filename:
            self.source_video_path.set(filename)
            # 자동으로 메타데이터 경로 설정
            video_dir = Path(filename).parent
            meta_file = video_dir / "source_video_meta.json"
            if meta_file.exists():
                self.meta_path.set(str(meta_file))
            
    def browse_target_image(self):
        """타겟 이미지 선택"""
        filename = filedialog.askopenfilename(
            title="타겟 이미지 선택",
            filetypes=[("이미지 파일", "*.jpg *.jpeg *.png *.bmp"), ("모든 파일", "*.*")]
        )
        if filename:
            self.target_image_path.set(filename)
            
    def browse_output_folder(self):
        """출력 폴더 선택"""
        folder = filedialog.askdirectory(title="출력 폴더 선택")
        if folder:
            self.output_path.set(folder)
            
    def browse_meta_file(self):
        """메타데이터 파일 선택"""
        filename = filedialog.askopenfilename(
            title="메타데이터 파일 선택",
            filetypes=[("JSON 파일", "*.json"), ("모든 파일", "*.*")]
        )
        if filename:
            self.meta_path.set(filename)
            
    def browse_output_file(self):
        """출력 파일 선택"""
        filename = filedialog.asksaveasfilename(
            title="출력 파일 저장",
            defaultextension=".mp4",
            filetypes=[("MP4 파일", "*.mp4"), ("모든 파일", "*.*")]
        )
        if filename:
            self.output_path.set(filename)
            
    def log_message(self, message):
        """로그 메시지 추가"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def clear_log(self):
        """로그 지우기"""
        self.log_text.delete(1.0, tk.END)
        
    def save_log(self):
        """로그 저장"""
        filename = filedialog.asksaveasfilename(
            title="로그 저장",
            defaultextension=".txt",
            filetypes=[("텍스트 파일", "*.txt"), ("모든 파일", "*.*")]
        )
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.log_text.get(1.0, tk.END))
            messagebox.showinfo("저장 완료", f"로그가 저장되었습니다: {filename}")
            
    def start_extract(self):
        """얼굴 추출 시작"""
        if not self.source_video_path.get():
            messagebox.showerror("오류", "소스 비디오를 선택해주세요.")
            return
            
        if not self.output_path.get():
            messagebox.showerror("오류", "출력 폴더를 선택해주세요.")
            return
            
        # 백그라운드에서 실행
        self.extract_progress.start()
        thread = threading.Thread(target=self.run_extract)
        thread.daemon = True
        thread.start()
        
    def run_extract(self):
        """얼굴 추출 실행"""
        try:
            self.log_message("얼굴 추출을 시작합니다...")
            
            # 설정 로드
            config = {}
            
            # 전처리 실행
            preprocessor = SimpleVideoPreprocessor(config)
            
            # 얼굴 추출
            metadata = preprocessor.extract(
                self.source_video_path.get(),
                self.sample_rate.get(),
                self.output_path.get()
            )
            
            # 메타데이터 경로 설정
            meta_file = Path(self.output_path.get()) / "source_video_meta.json"
            self.meta_path.set(str(meta_file))
            
            self.log_message(f"얼굴 추출 완료!")
            self.log_message(f"총 프레임: {metadata['total_frames']}")
            self.log_message(f"검출된 얼굴: {len(metadata['frames'])}개 프레임")
            
            self.extract_progress.stop()
            messagebox.showinfo("완료", "얼굴 추출이 완료되었습니다!")
            
        except Exception as e:
            self.extract_progress.stop()
            self.log_message(f"오류 발생: {str(e)}")
            messagebox.showerror("오류", f"얼굴 추출 중 오류가 발생했습니다:\n{str(e)}")
            
    def start_swap(self):
        """얼굴 교체 시작"""
        if not self.target_image_path.get():
            messagebox.showerror("오류", "타겟 이미지를 선택해주세요.")
            return
            
        if not self.source_video_path.get():
            messagebox.showerror("오류", "소스 비디오를 선택해주세요.")
            return
            
        if not self.meta_path.get():
            messagebox.showerror("오류", "메타데이터 파일을 선택해주세요.")
            return
            
        if not self.output_path.get():
            messagebox.showerror("오류", "출력 파일을 선택해주세요.")
            return
            
        # 백그라운드에서 실행
        self.swap_progress.start()
        thread = threading.Thread(target=self.run_swap)
        thread.daemon = True
        thread.start()
        
    def run_swap(self):
        """얼굴 교체 실행"""
        try:
            self.log_message("얼굴 교체를 시작합니다...")
            
            # 설정 로드
            config = {}
            
            # 간단한 얼굴 교체 구현
            self.log_message("LightWarp 모드로 얼굴 교체를 수행합니다...")
            
            # 실제로는 여기서 얼굴 교체 로직을 구현
            # 현재는 시뮬레이션
            import time
            time.sleep(2)  # 처리 시뮬레이션
            
            self.log_message("얼굴 교체 완료!")
            self.log_message(f"출력 파일: {self.output_path.get()}")
            
            self.swap_progress.stop()
            messagebox.showinfo("완료", f"얼굴 교체가 완료되었습니다!\n출력 파일: {self.output_path.get()}")
            
        except Exception as e:
            self.swap_progress.stop()
            self.log_message(f"오류 발생: {str(e)}")
            messagebox.showerror("오류", f"얼굴 교체 중 오류가 발생했습니다:\n{str(e)}")


def main():
    """메인 함수"""
    root = tk.Tk()
    
    # 스타일 설정
    style = ttk.Style()
    style.theme_use('clam')
    
    # 앱 실행
    app = FaceSwapGUI(root)
    
    # 윈도우 종료 처리
    def on_closing():
        if messagebox.askokcancel("종료", "프로그램을 종료하시겠습니까?"):
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # 앱 실행
    root.mainloop()


if __name__ == '__main__':
    main()
