"""Decode preset renders, measure expression/audio preservation, and export visual evidence."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import correlate, correlation_lags

from src.local.faces import Landmarker, OVAL
from src.local.media import CREATE_NO_WINDOW, ffmpeg, video_info, write_json
from src.local.tracking import face_in_region


def expression(points):
    return [float(np.linalg.norm(points[a]-points[b])/max(.001, np.linalg.norm(points[c]-points[d])))
            for a,b,c,d in ((159,145,33,133),(386,374,362,263),(13,14,78,308))]


def pcm(path):
    result = subprocess.run([ffmpeg(),'-v','error','-i',str(path),'-vn','-ac','1','-ar','16000',
                             '-f','f32le','-'], capture_output=True, check=True, creationflags=CREATE_NO_WINDOW)
    return np.frombuffer(result.stdout,np.float32)


def evaluate(source, result, model, roi):
    original = cv2.VideoCapture(str(source)); output = cv2.VideoCapture(str(result))
    samples, patches, changes = [], {}, []
    index = 0
    try:
        while True:
            a, frame = original.read(); b, swapped = output.read()
            if not a or not b:
                if a != b:
                    raise RuntimeError('Decoded input/output frame counts differ')
                break
            if index % 3 == 0:
                p = face_in_region(frame, roi, model)
                q = face_in_region(swapped, roi, model)
                if p is not None and q is not None:
                    samples.append({'frame':index,'original':expression(p),'result':expression(q)})
                    mask = np.zeros(frame.shape[:2], np.uint8)
                    cv2.fillConvexPoly(mask, cv2.convexHull(np.rint(p[OVAL]).astype(np.int32)), 1)
                    delta = np.abs(frame.astype(np.float32)-swapped)
                    changes.append(float(delta[mask > 0].mean()))
            if index % 30 == 0:
                x,y,w,h = roi
                patches[index] = (frame[y:y+h,x:x+w], swapped[y:y+h,x:x+w])
            index += 1
    finally:
        original.release(); output.release()
    values = np.array([[s['original'],s['result']] for s in samples])
    correlations = {name: float(np.corrcoef(values[:,0,i],values[:,1,i])[0,1])
                    for i,name in enumerate(('left_eye','right_eye','mouth'))} if samples else {}
    x,y = pcm(source),pcm(result)
    x=x-x.mean();y=y-y.mean()
    cc=correlate(y,x,mode='full',method='fft');lags=correlation_lags(len(y),len(x))
    eligible=np.abs(lags)<=3200
    lag=int(lags[eligible][np.argmax(cc[eligible])])
    audio={'lag_ms':lag/16,'correlation':float(cc[eligible].max()/np.sqrt(np.sum(x*x)*np.sum(y*y)))}
    decode=subprocess.run([ffmpeg(),'-v','error','-i',str(result),'-f','null','-'],capture_output=True,creationflags=CREATE_NO_WINDOW)
    metadata=subprocess.run([ffmpeg(),'-hide_banner','-i',str(result)],capture_output=True,creationflags=CREATE_NO_WINDOW).stderr.decode('utf-8','replace')
    stats={'info':video_info(result),'decoded_frames':index,'paired_detections':len(samples),
           'sampled_frames':(index+2)//3,'expression_correlations':correlations,
           'expression_threshold_pass':len(samples)==(index+2)//3 and bool(correlations) and all(np.isfinite(v) and v>=.95 for v in correlations.values()),
           'audio':audio,'full_decode_pass':decode.returncode==0 and not decode.stderr.strip(),
           'streams':[s.strip() for s in metadata.splitlines() if 'Stream #' in s],
           'mean_face_pixel_change':float(np.mean(changes)) if changes else None,'samples':samples}
    return stats, patches


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--folder',required=True)
    parser.add_argument('--roi',nargs=4,type=int,required=True,metavar=('X','Y','W','H'))
    args=parser.parse_args()
    folder=Path(args.folder);names=('legacy','balanced','strong')
    model=Landmarker(); reports={}; patches={}
    try:
        for name in names:
            reports[name], patches[name]=evaluate(Path(args.source),folder/f'{name}.mp4',model,args.roi)
            print(name,json.dumps({k:v for k,v in reports[name].items() if k!='samples'}),flush=True)
    finally:
        model.close()
    write_json(folder/'quality_metrics.json',{'presets':reports,'limitations':[
        'Same landmark model family as compositor; expression correlation is not an identity or naturalness score.',
        'Audio lag compares output to original waveform, not original audiovisual lip sync.',
        'One frontal 10-second clip does not validate profiles, occlusions, or all-frame perceptual quality.']})
    frames=sorted(patches['legacy'])
    sheet=Image.new('RGB',(1200,len(frames)*300),'#18202d');draw=ImageDraw.Draw(sheet)
    with Image.open(folder/'source_face.png') as image:
        reference=Image.new('RGBA',image.size,'#d9d9d9');reference.alpha_composite(image.convert('RGBA'))
        reference=reference.convert('RGB')
    for row,index in enumerate(frames):
        panels=[reference,Image.fromarray(cv2.cvtColor(patches['legacy'][index][0],cv2.COLOR_BGR2RGB))]
        panels += [Image.fromarray(cv2.cvtColor(patches[name][index][1],cv2.COLOR_BGR2RGB)) for name in names]
        for col,(title,panel) in enumerate(zip(('UPLOADED FACE','ORIGINAL')+names,panels)):
            panel=panel.copy();panel.thumbnail((230,265))
            sheet.paste(panel,(col*240+(240-panel.width)//2,row*300+30))
            draw.text((col*240+8,row*300+8),f'{title} {index/30:.1f}s',fill='white')
    sheet.save(folder/'rendered_contact_sheet.jpg',quality=95)
    selected=Image.new('RGB',(1200,1200),'#18202d')
    for row,second in enumerate((0,2,6,8)):
        selected.paste(sheet.crop((0,second*300,1200,(second+1)*300)),(0,row*300))
    selected.save(folder/'rendered_keyframes.jpg',quality=95)
    decode=subprocess.run([ffmpeg(),'-v','error','-i',str(folder/'comparison.mp4'),'-f','null','-'],capture_output=True,creationflags=CREATE_NO_WINDOW)
    if decode.returncode:
        raise RuntimeError(decode.stderr.decode('utf-8','replace'))


if __name__=='__main__':
    main()
