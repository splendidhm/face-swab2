"""
얼굴 검출 모듈
"""
import cv2
import numpy as np
import mediapipe as mp
import dlib
from typing import List, Tuple, Optional, Dict
from pathlib import Path


class FaceDetector:
    """얼굴 검출 클래스"""
    
    def __init__(self, method: str = 'mediapipe', confidence_threshold: float = 0.5):
        """
        Args:
            method: 검출 방법 ('mediapipe', 'dlib', 'opencv', 'mtcnn')
            confidence_threshold: 신뢰도 임계값
        """
        self.method = method
        self.confidence_threshold = confidence_threshold
        
        if method == 'mediapipe':
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=confidence_threshold
            )
        elif method == 'dlib':
            # dlib 얼굴 검출기 초기화
            self.detector = dlib.get_frontal_face_detector()
        elif method == 'opencv':
            # OpenCV Haar Cascade 초기화
            cascade_path = Path(__file__).parent.parent.parent / "haarcascade_frontalface_default.xml"
            if cascade_path.exists():
                self.face_cascade = cv2.CascadeClassifier(str(cascade_path))
            else:
                raise FileNotFoundError("Haar cascade 파일을 찾을 수 없습니다")
        elif method == 'mtcnn':
            try:
                from mtcnn import MTCNN
                self.mtcnn = MTCNN(min_face_size=20, confidence_threshold=confidence_threshold)
            except ImportError:
                raise ImportError("MTCNN을 설치해주세요: pip install mtcnn")
        else:
            raise ValueError(f"지원하지 않는 검출 방법: {method}")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        이미지에서 얼굴 검출
        
        Args:
            image: 입력 이미지 (BGR)
            
        Returns:
            faces: 검출된 얼굴 정보 리스트
        """
        if self.method == 'mediapipe':
            return self._detect_mediapipe(image)
        elif self.method == 'dlib':
            return self._detect_dlib(image)
        elif self.method == 'opencv':
            return self._detect_opencv(image)
        elif self.method == 'mtcnn':
            return self._detect_mtcnn(image)
    
    def _detect_mediapipe(self, image: np.ndarray) -> List[Dict]:
        """MediaPipe를 사용한 얼굴 검출"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        faces = []
        if results.detections:
            h, w = image.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                confidence = detection.score[0]
                if confidence >= self.confidence_threshold:
                    faces.append({
                        'bbox': (x, y, width, height),
                        'confidence': confidence,
                        'landmarks': None  # MediaPipe face detection은 랜드마크 제공 안함
                    })
        
        return faces
    
    def _detect_dlib(self, image: np.ndarray) -> List[Dict]:
        """dlib을 사용한 얼굴 검출"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = self.detector(gray)
        
        faces = []
        for detection in detections:
            x = detection.left()
            y = detection.top()
            width = detection.width()
            height = detection.height()
            
            faces.append({
                'bbox': (x, y, width, height),
                'confidence': 1.0,  # dlib은 신뢰도 제공 안함
                'landmarks': None
            })
        
        return faces
    
    def _detect_opencv(self, image: np.ndarray) -> List[Dict]:
        """OpenCV Haar Cascade를 사용한 얼굴 검출"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        faces = []
        for (x, y, w, h) in detections:
            faces.append({
                'bbox': (x, y, w, h),
                'confidence': 1.0,  # Haar cascade는 신뢰도 제공 안함
                'landmarks': None
            })
        
        return faces
    
    def _detect_mtcnn(self, image: np.ndarray) -> List[Dict]:
        """MTCNN을 사용한 얼굴 검출"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self.mtcnn.detect_faces(rgb_image)
        
        faces = []
        for detection in detections:
            if detection['confidence'] >= self.confidence_threshold:
                bbox = detection['box']
                x, y, w, h = bbox
                
                faces.append({
                    'bbox': (x, y, w, h),
                    'confidence': detection['confidence'],
                    'landmarks': detection.get('keypoints', None)
                })
        
        return faces
    
    def detect_largest_face(self, image: np.ndarray) -> Optional[Dict]:
        """
        가장 큰 얼굴 검출
        
        Args:
            image: 입력 이미지
            
        Returns:
            face: 가장 큰 얼굴 정보 또는 None
        """
        faces = self.detect_faces(image)
        
        if not faces:
            return None
        
        # 가장 큰 얼굴 찾기
        largest_face = max(faces, key=lambda f: f['bbox'][2] * f['bbox'][3])
        return largest_face
    
    def crop_face(self, image: np.ndarray, face_info: Dict, 
                 margin: float = 0.2) -> np.ndarray:
        """
        얼굴 영역 크롭
        
        Args:
            image: 입력 이미지
            face_info: 얼굴 정보
            margin: 여백 비율
            
        Returns:
            cropped_face: 크롭된 얼굴 이미지
        """
        x, y, w, h = face_info['bbox']
        
        # 여백 추가
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + w + margin_x)
        y2 = min(image.shape[0], y + h + margin_y)
        
        return image[y1:y2, x1:x2]
    
    def visualize_detections(self, image: np.ndarray, faces: List[Dict], 
                           color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        검출된 얼굴을 이미지에 시각화
        
        Args:
            image: 입력 이미지
            faces: 검출된 얼굴 리스트
            color: 색상 (BGR)
            
        Returns:
            vis_image: 시각화된 이미지
        """
        vis_image = image.copy()
        
        for face in faces:
            x, y, w, h = face['bbox']
            confidence = face.get('confidence', 1.0)
            
            # 바운딩 박스 그리기
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            
            # 신뢰도 텍스트
            if confidence < 1.0:
                text = f"{confidence:.2f}"
                cv2.putText(vis_image, text, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis_image
    
    def batch_detect(self, image_paths: List[str], output_dir: str) -> Dict[str, List[Dict]]:
        """
        여러 이미지에서 배치 얼굴 검출
        
        Args:
            image_paths: 이미지 파일 경로 리스트
            output_dir: 출력 디렉토리
            
        Returns:
            results: 검출 결과 딕셔너리
        """
        from tqdm import tqdm
        
        results = {}
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for image_path in tqdm(image_paths, desc="얼굴 검출 중"):
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            faces = self.detect_faces(image)
            results[image_path] = faces
            
            # 시각화된 이미지 저장
            if faces:
                vis_image = self.visualize_detections(image, faces)
                output_file = output_path / f"detected_{Path(image_path).name}"
                cv2.imwrite(str(output_file), vis_image)
        
        return results

