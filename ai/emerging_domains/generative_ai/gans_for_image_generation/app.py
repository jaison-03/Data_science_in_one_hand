"""Train a compact NumPy GAN that generates synthetic grayscale digits."""
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.datasets import load_digits

if not st.session_state.get("_embedded_ai_project"):
    st.set_page_config(page_title="Synthetic Image Generator",page_icon="🎨",layout="wide")
st.markdown("<style>.block-container{max-width:1100px;padding-top:2rem}</style>",unsafe_allow_html=True)
st.title("Synthetic Image Generator")
st.caption("Train a small fully connected GAN on the built-in 8×8 digits dataset and inspect generated samples.")
digits=load_digits()
limit=st.sidebar.slider("Training examples",200,1400,600,100)
epochs=st.sidebar.slider("Training epochs",5,100,30,5)
width=st.sidebar.slider("Hidden units",24,128,64,8)
seed=st.sidebar.number_input("Random seed",0,9999,23)
latent_size=16
real_images=digits.images.reshape(-1,64).astype(np.float32)/8.0-1.0

def sigmoid(values):
    return 1.0/(1.0+np.exp(-np.clip(values,-30,30)))

@st.cache_resource
def train_gan(count,hidden,rounds,random_seed):
    rng=np.random.default_rng(random_seed)
    real=real_images[:count]
    scale=.06
    params={
        "wg1":rng.normal(0,scale,(latent_size,hidden)),"bg1":np.zeros((1,hidden)),
        "wg2":rng.normal(0,scale,(hidden,64)),"bg2":np.zeros((1,64)),
        "wd1":rng.normal(0,scale,(64,hidden)),"bd1":np.zeros((1,hidden)),
        "wd2":rng.normal(0,scale,(hidden,1)),"bd2":np.zeros((1,1)),
    }
    first={name:np.zeros_like(value) for name,value in params.items()}
    second={name:np.zeros_like(value) for name,value in params.items()}
    step=0; losses=[]
    def generate(z):
        g1=np.tanh(z@params["wg1"]+params["bg1"])
        image=np.tanh(g1@params["wg2"]+params["bg2"])
        return image,g1
    def discriminate(image):
        d1=image@params["wd1"]+params["bd1"]
        h=np.maximum(d1,0)
        probability=sigmoid(h@params["wd2"]+params["bd2"])
        return probability,d1,h
    def update(gradients,rate=.001):
        nonlocal step
        step+=1
        for name,gradient in gradients.items():
            first[name]=.9*first[name]+.1*gradient
            second[name]=.999*second[name]+.001*gradient*gradient
            mhat=first[name]/(1-.9**step);vhat=second[name]/(1-.999**step)
            params[name]-=rate*mhat/(np.sqrt(vhat)+1e-8)
    batch_size=64
    for epoch in range(rounds):
        epoch_loss=[]
        for offset in range(0,count,batch_size):
            batch=real[offset:offset+batch_size]
            size=len(batch)
            # Train the discriminator on real and generated examples.
            z=rng.normal(size=(size,latent_size));fake,_=generate(z)
            real_prob,real_pre,real_hidden=discriminate(batch)
            fake_prob,fake_pre,fake_hidden=discriminate(fake)
            real_delta=(real_prob-1)/size;fake_delta=fake_prob/size
            gradients={}
            gradients["wd2"]=real_hidden.T@real_delta+fake_hidden.T@fake_delta
            gradients["bd2"]=real_delta.sum(axis=0,keepdims=True)+fake_delta.sum(axis=0,keepdims=True)
            real_h_delta=(real_delta@params["wd2"].T)*(real_pre>0)
            fake_h_delta=(fake_delta@params["wd2"].T)*(fake_pre>0)
            gradients["wd1"]=batch.T@real_h_delta+fake.T@fake_h_delta
            gradients["bd1"]=real_h_delta.sum(axis=0,keepdims=True)+fake_h_delta.sum(axis=0,keepdims=True)
            update(gradients)
            # Train the generator to make discriminator predictions positive.
            z=rng.normal(size=(size,latent_size));fake,g_hidden=generate(z)
            probability,d_pre,d_hidden=discriminate(fake)
            delta=(probability-1)/size
            grad_d_hidden=(delta@params["wd2"].T)*(d_pre>0)
            grad_image=grad_d_hidden@params["wd1"].T
            grad_g_out=grad_image*(1-fake*fake)
            grad_g_hidden=(grad_g_out@params["wg2"].T)*(1-g_hidden*g_hidden)
            update({
                "wg2":g_hidden.T@grad_g_out,"bg2":grad_g_out.sum(axis=0,keepdims=True),
                "wg1":z.T@grad_g_hidden,"bg1":grad_g_hidden.sum(axis=0,keepdims=True),
            })
            epoch_loss.append(float(-np.mean(np.log(real_prob+1e-8))-np.mean(np.log(1-fake_prob+1e-8))))
        losses.append(float(np.mean(epoch_loss)))
    return params,losses

with st.spinner(f"Training the GAN for {epochs} epochs..."):
    parameters,history=train_gan(limit,width,epochs,int(seed))
rng=np.random.default_rng(int(seed)+1)
z=rng.normal(size=(16,latent_size))
generated_hidden=np.tanh(z@parameters["wg1"]+parameters["bg1"])
generated=np.clip(np.tanh(generated_hidden@parameters["wg2"]+parameters["bg2"]).reshape(-1,8,8),-1,1)
generated=(generated+1)/2
st.subheader("Generated samples")
fig=px.imshow(np.concatenate([np.concatenate(generated[i:i+4],axis=1) for i in range(0,16,4)],axis=0),color_continuous_scale="gray",zmin=0,zmax=1)
fig.update_layout(height=480,coloraxis_showscale=False); fig.update_xaxes(showticklabels=False); fig.update_yaxes(showticklabels=False)
st.plotly_chart(fig,use_container_width=True)
st.plotly_chart(px.line(x=range(1,len(history)+1),y=history,labels={"x":"Epoch","y":"Discriminator BCE loss"},title="Adversarial training loss"),use_container_width=True)
st.warning("This is a small educational GAN trained on 8×8 grayscale digits; mode collapse and blurry outputs are expected. It is not a production-quality image model.")
