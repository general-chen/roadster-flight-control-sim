"""Continuous plotted target; the roll segment is an interpolated visual guide.
It is not an enforced position command or a controller benchmark.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from simulate import ROLL_START,ROLL_END,reference,DIR,MASS,GRAVITY

def roll_bridge(A):
 j=int(np.flatnonzero(A[:,0]>=ROLL_END)[0]);T=ROLL_END-ROLL_START
 p0,v0,a0=reference(ROLL_START);p1=A[j,24:27];v1=A[j,27:30]
 R=Rotation.from_quat(A[j,7:11]).as_matrix()
 a1=R@(DIR.T@A[j,14:24])/MASS-np.array([0.,0.,GRAVITY]);a1[0]=0.
 c=np.zeros((6,3));c[:3]=[p0,v0,a0/2]
 mat=np.array([[T**3,T**4,T**5],[3*T*T,4*T**3,5*T**4],[6*T,12*T*T,20*T**3]])
 c[3:]=np.linalg.solve(mat,[p1-(c[0]+c[1]*T+c[2]*T*T),v1-c[1]-2*c[2]*T,a1-2*c[2]])
 return c

def display_reference(A):
 t=A[:,0];guide=(t>=ROLL_START)&(t<ROLL_END);ref=A[:,24:27].copy();u=t[guide]-ROLL_START
 c=roll_bridge(A);ref[guide]=sum(u[:,None]**k*c[k] for k in range(6))
 return ref,guide
