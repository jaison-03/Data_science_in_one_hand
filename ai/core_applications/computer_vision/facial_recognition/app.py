"""Consent-based face detection and within-session image matching demo."""
import numpy as np
import streamlit as st
from PIL import Image,ImageDraw
try:
    import cv2
except ImportError:
    cv2=None

if not st.session_state.get("_embedded_ai_project"):
    st.set_page_config(page_title="Face Detection and Matching",page_icon="🙂",layout="wide")
st.markdown("<style>.block-container{max-width:1100px;padding-top:2rem}</style>",unsafe_allow_html=True)
st.title("Face Detection & Image Matching")
st.caption("Detect faces and compare multiple consented images within this session. No identities are inferred or sent to a service.")
st.warning("Use only images you have permission to process. This is a basic appearance matcher, not identity verification.")
files=st.file_uploader("Choose two or more face images",type=["png","jpg","jpeg"],accept_multiple_files=True)
if len(files)<2: st.info("Upload at least two images to compare."); st.stop()
if cv2 is None:
    st.error("Install the root requirements to enable OpenCV face detection (`opencv-python-headless`).")
    st.stop()
cascade_path=cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
cascade=cv2.CascadeClassifier(cascade_path)
if cascade.empty(): st.error("OpenCV face detector could not be loaded. Reinstall opencv-python-headless."); st.stop()
records=[]
for file in files[:10]:
    try:
        image=Image.open(file).convert("RGB")
        if image.width*image.height>12_000_000: st.warning(f"Skipped {file.name}: image too large."); continue
        image.thumbnail((900,900))
        array=np.asarray(image)
        gray=cv2.cvtColor(array,cv2.COLOR_RGB2GRAY)
        boxes=cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(40,40))
        shown=image.copy();draw=ImageDraw.Draw(shown)
        features=[]
        for x,y,w,h in boxes:
            draw.rectangle((x,y,x+w,y+h),outline="red",width=3)
            face=cv2.resize(gray[y:y+h,x:x+w],(64,64)).astype(np.float32)
            features.append(face)
        records.append((file.name,shown,features))
    except (OSError,ValueError) as error:
        st.warning(f"Could not process {file.name}: {error}")
if records:
    cols=st.columns(min(3,len(records)))
    for index,(name,image,faces) in enumerate(records):
        with cols[index%len(cols)]:
            st.image(image,caption=f"{name} · {len(faces)} face(s)",use_container_width=True)
matches=[]
for i in range(len(records)):
    for j in range(i+1,len(records)):
        left=records[i][2];right=records[j][2]
        if not left or not right: continue
        scores=[]
        for a in left:
            for b in right:
                a=(a-a.mean())/(a.std()+1e-6);b=(b-b.mean())/(b.std()+1e-6)
                scores.append(float(np.mean(a*b)))
        matches.append({"Image A":records[i][0],"Image B":records[j][0],"Best normalized appearance similarity":round(max(scores),3)})
if matches:
    st.subheader("Within-session appearance comparison")
    st.dataframe(matches,hide_index=True,use_container_width=True)
    st.caption("Similarity is based on normalized grayscale pixels; pose, lighting, and expression strongly affect results. It is not reliable biometric identification.")
else:
    st.info("At least one detectable face is needed in each image to compare appearances.")
