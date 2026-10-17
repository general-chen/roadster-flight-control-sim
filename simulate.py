"""Roadster-inspired 3D PID test. Assumed hardware, geometric PD + position PID.
Run: python simulate.py    (Python 3.10+, numpy, scipy, matplotlib, pillow)
No Tesla controller or calibrated vehicle model is claimed.
"""
from pathlib import Path
import argparse, json
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT=Path(__file__).resolve().parent
ROLL_START=1.0; ROLL_DURATION=1.5; ROLL_END=ROLL_START+ROLL_DURATION; RECOVERY_DURATION=4.0
LANDING_DURATION=1.6; TOUCHDOWN_HEIGHT=.548
BOOST_ACCEL=5.8; INITIAL_YAW_DEG=2.5
MASS=2000.; GRAVITY=9.81; INERTIA=np.array([900.,3600.,4000.])
# Body axes: x forward, y left, z up. Locations measured from COM, metres.
LOC=np.array([[1.6,.8,-.3],[1.6,-.8,-.3],[-1.6,.8,-.3],[-1.6,-.8,-.3],
              [1.9,1.02,-.1],[1.9,-1.02,-.1],[-2.,.7,0],[-2.,.23,0],[-2.,-.23,0],[-2.,-.7,0]])
DIR=np.array([[0,0,1]]*4+[[0,-1,0],[0,1,0]]+[[1,0,0]]*4,dtype=float)
B=np.vstack((DIR.T,np.cross(LOC,DIR).T))
CORNERS=np.array([[x,y,z] for x in [-2.25,2.25] for y in [-1.15,1.15] for z in [-.55,.65]])
FACES=[[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]
# Row width is flight-path extent. Parked vehicles face sideways to flight path.
OBSTACLES=[]
edge=10.
for name,width,height,color in [('Car A',2.,1.5,'#a44b58'),('Car B',2.,1.6,'#596fa0'),('Truck',2.6,4.1,'#acbac9'),('Pickup',2.4,1.8,'#68737d'),('Car C',2.,1.5,'#bf9c4c')]:
 OBSTACLES.append(dict(name=name,lo=[edge,-3.5,0.],hi=[edge+width,3.5,height],color=color));edge+=width+.4

def smooth(t,start,duration,a,b):
 u=np.clip((t-start)/duration,0,1); d=b-a
 return a+d*(10*u**3-15*u**4+6*u**5),d*(30*u*u-60*u**3+30*u**4)/duration,d*(60*u-180*u*u+120*u**3)/duration**2

def reference(t):
 # Boost reference; the roll uses attitude tracking while translation coasts.
 z=.55+.5*BOOST_ACCEL*t*t;vz=BOOST_ACCEL*t
 return np.array([9.*t,0,z]),np.array([9.,0,vz]),np.array([0.,0,BOOST_ACCEL])

def quintic_coefficients(start,velocity,acceleration,goal,T):
 # C2 endpoint construction: match measured p/v/a, finish at zero v/a.
 c=np.zeros(6);c[:3]=[start,velocity,.5*acceleration]
 matrix=np.array([[T**3,T**4,T**5],[3*T*T,4*T**3,5*T**4],[6*T,12*T*T,20*T**3]])
 c[3:]=np.linalg.solve(matrix,[goal-sum(c[k]*T**k for k in range(3)),-c[1]-2*c[2]*T,-2*c[2]])
 return c

def recovery_coefficients(position,velocity,acceleration):
 # Constant positive forward velocity; the vehicle never stops in the air.
 return dict(x0=position[0],vx=velocity[0],
  y=quintic_coefficients(position[1],velocity[1],acceleration[1],0.,RECOVERY_DURATION),
  z=quintic_coefficients(position[2],velocity[2],acceleration[2],TOUCHDOWN_HEIGHT,LANDING_DURATION))

def sample_polynomial(c,t,T):
 u=np.clip(t,0,T)
 return sum(c[k]*u**k for k in range(6)),sum(k*c[k]*u**(k-1) for k in range(1,6)),sum(k*(k-1)*c[k]*u**(k-2) for k in range(2,6))

def sample_recovery(t,c):
 u=t-ROLL_END;y=sample_polynomial(c['y'],u,RECOVERY_DURATION);z=sample_polynomial(c['z'],u,LANDING_DURATION)
 return np.array([c['x0']+c['vx']*u,y[0],z[0]]),np.array([c['vx'],y[1],z[1]]),np.array([0.,y[2],z[2]])

def box_vertices(lo,hi):
 return np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])

def overlap_obb_aabb(p,R,lo,hi):
 # Separating-axis theorem, 15 axes. Full box, not just COM/corners.
 c=p+R@np.array([0,0,.05]);half=np.array([2.25,1.15,.6]);oc=(lo+hi)/2;oh=(hi-lo)/2
 for axis in [*R.T,*np.eye(3),*[np.cross(R[:,i],np.eye(3)[j]) for i in range(3) for j in range(3)]]:
  if np.linalg.norm(axis)<1e-10:continue
  if abs((c-oc)@axis)>np.sum(half*np.abs(R.T@axis))+np.sum(oh*np.abs(axis))+1e-9:return False
 return True

def run(dt=.0025,disturbance=False):
 p=np.array([0.,0.,.550001]);v=np.array([9.,0.,0.]);R=Rotation.from_euler('z',INITIAL_YAW_DEG,degrees=True).as_matrix();w=np.zeros(3);integ=np.zeros(3);f=np.zeros(10)
 if disturbance:R=Rotation.from_euler('z',INITIAL_YAW_DEG+1.,degrees=True).as_matrix() # yaw only; no initial floor penetration
 recovery=None;descent_start=ROLL_END;Rd_previous=np.eye(3);
 logs=[]; minclear=np.inf; collision=False; airborne=False; touchdown=None
 for k in range(int(13/dt)+1):
  t=k*dt;pr,vr,ar=reference(t)
  if t>=ROLL_END:
   if recovery is None:
    handoff_wind=np.array([0.,700.*np.sin(1.7*t),0.]) if disturbance else np.zeros(3)
    handoff_accel=(R@(DIR.T@f)+handoff_wind)/MASS-np.array([0.,0.,GRAVITY])
    recovery=recovery_coefficients(p.copy(),v.copy(),handoff_accel)
   pr,vr,ar=sample_recovery(t,recovery)
  elif t>=ROLL_START:
   # Ballistic reference is only diagnostic; altitude PID is suspended during roll.
   u=t-ROLL_START;pr=np.array([9*t,0.,.55+.5*BOOST_ACCEL*ROLL_START**2+BOOST_ACCEL*ROLL_START*u-.5*GRAVITY*u*u]);vr=np.array([9.,0.,BOOST_ACCEL*ROLL_START-GRAVITY*u]);ar=np.array([0.,0.,-GRAVITY])
  verts=p+CORNERS@R.T
  # Simplified unilateral launch-pad support during initial valve spool-up.
  if not airborne and verts[:,2].min()<0:
   p[2]-=verts[:,2].min();v[2]=max(v[2],0.);verts=p+CORNERS@R.T
  for ob in OBSTACLES:
   lo=np.array(ob['lo']);hi=np.array(ob['hi'])
   if np.all(verts.max(axis=0)>=lo) and np.all(verts.min(axis=0)<=hi):
    collision |= overlap_obb_aabb(p,R,lo,hi)
   if verts[:,0].max()>=lo[0] and verts[:,0].min()<=hi[0] and verts[:,1].max()>=lo[1] and verts[:,1].min()<=hi[1]:
    minclear=min(minclear,float(verts[:,2].min()-hi[2]))
  logs.append(np.r_[t,p,v,Rotation.from_matrix(R).as_quat(),w,f,pr,vr])
  if verts[:,2].min()>.10:airborne=True
  if airborne and verts[:,2].min()<=0 and v[2]<0:
   touchdown={'time_s':float(t),'position_m':p.tolist(),'vertical_speed_m_s':float(v[2]),'horizontal_speed_m_s':float(np.linalg.norm(v[:2])), 'attitude_deg':Rotation.from_matrix(R).as_euler('xyz',degrees=True).tolist(),'angular_speed_rad_s':float(np.linalg.norm(w)),'vertical_contact_point_speed_m_s':float((v+R@np.cross(w,CORNERS[np.argmin(verts[:,2])]))[2]),'max_floor_penetration_at_detection_m':float(-verts[:,2].min())};logs[-1]=np.r_[logs[-1],Rotation.from_matrix(Rd_previous).as_quat()];break
  err=pr-p
  if not (ROLL_START<=t<ROLL_END):integ=np.clip(integ+dt*err,-.4,.4)
  acc=ar+np.array([1.8,1.8,16.])*err+np.array([2.5,2.5,8.])*(vr-v)+np.array([.05,.05,.2])*integ
  target=acc+np.array([0,0,GRAVITY])
  zb=target/np.linalg.norm(target)
  yb=np.cross(zb,[1.,0,0]);yb/=np.linalg.norm(yb);xb=np.cross(yb,zb);Rd=np.column_stack([xb,yb,zb])
  roll_active=ROLL_START<=t<ROLL_END
  rate_ref=np.zeros(3);accel_ref=np.zeros(3)
  if roll_active:
   phi,rate,alpha=smooth(t,ROLL_START,ROLL_DURATION,0.,2*np.pi)
   Rd=Rotation.from_rotvec([phi,0.,0.]).as_matrix()
   rate_ref=R.T@Rd@np.array([rate,0,0]);accel_ref=R.T@Rd@np.array([alpha,0,0])
  if not roll_active:
   desired_rate=Rotation.from_matrix(Rd_previous.T@Rd).as_rotvec()/dt
   rate_ref=R.T@Rd@np.clip(desired_rate,-5.,5.)
  Rd_previous=Rd.copy()
  logs[-1]=np.r_[logs[-1],Rotation.from_matrix(Rd).as_quat()]
  E=.5*(Rd.T@R-R.T@Rd);e=np.array([E[2,1],E[0,2],E[1,0]])
  tau=-np.array([60000.,180000.,80000.])*e-np.array([14000.,50000.,32000.])*(w-rate_ref)+np.cross(w,INERTIA*w)+INERTIA*accel_ref
  # Exact allocation for vertical force / roll / pitch of symmetric underside jets.
  lift=np.clip(MASS*target@R[:,2],0,40000.)
  if roll_active:
   # Minimum nonnegative collective needed to generate the demanded moments.
   # This retains actual thrust and its translational effect even while inverted.
   lift=np.clip(abs(tau[0])/.8+abs(tau[1])/1.6,0,40000.)
  fcmd=np.zeros(10)
  fcmd[:4]=np.linalg.solve(B[[2,3,4],:4]@B[[2,3,4],:4].T,np.array([lift,tau[0],tau[1]]))@B[[2,3,4],:4]
  # Lateral pair produces yaw AND lateral force; real force coupling stays in plant.
  side=np.clip(tau[2]/1.9,-2000,2000)
  fcmd[4]=max(-side,0);fcmd[5]=max(side,0)
  fcmd[:4]=np.clip(fcmd[:4],0,10000.)
  # Rear thrusters physically modelled, deliberately inactive in this first task.
  f+=(1-np.exp(-dt/.03))*(fcmd-f)
  force=R@(DIR.T@f);moment=np.cross(LOC,DIR*f[:,None]).sum(axis=0)
  wind=np.array([0.,700.*np.sin(1.7*t),0.]) if disturbance else np.zeros(3)
  a=(force+wind)/MASS-np.array([0,0,GRAVITY])
  wd=(moment-np.cross(w,INERTIA*w))/INERTIA
  p+=v*dt+.5*a*dt*dt;v+=a*dt
  R=R@Rotation.from_rotvec((w+.5*wd*dt)*dt).as_matrix();w+=wd*dt
 A=np.array(logs)
 metrics={'mass_kg':MASS,'inertia_kg_m2':INERTIA.tolist(),'dt_s':dt,'collision':bool(collision),'minimum_conservative_vertical_clearance_m':float(minclear),'touchdown':touchdown,'total_impulse_N_s':float(np.trapezoid(A[:,14:24].sum(axis=1),A[:,0])),'position_RMSE_m':float(np.sqrt(np.mean(np.sum((A[:,1:4]-A[:,24:27])**2,axis=1)))),'gas_supply':'unlimited; no depletion or pressure model','mass_model':'constant','initial_forward_speed_m_s':9.,'rear_thrusters_active':False,'success':bool(touchdown and not collision and abs(touchdown['horizontal_speed_m_s']-9.)<.5 and abs(touchdown['vertical_contact_point_speed_m_s'])<.5 and max(abs(np.array(touchdown['attitude_deg'])))<5 and touchdown['angular_speed_rad_s']<.1 and p[0]-2.25>22.6)}
 roll=np.unwrap(Rotation.from_quat(A[:,7:11]).as_euler('xyz')[:,0])
 metrics['net_roll_degrees']=float(np.rad2deg(roll[-1]-roll[0]));metrics['maximum_height_m']=float(A[:,3].max())
 metrics['rear_body_gap_to_row_at_contact_m']=float((p+CORNERS@R.T)[:,0].min()-22.6) if touchdown else None
 metrics['front_lateral_peak_thrust_N']=A[:,18:20].max(axis=0).tolist()
 metrics['initial_yaw_deg']=INITIAL_YAW_DEG
 metrics['maximum_absolute_heading_yaw_deg']=float(abs(Rotation.from_quat(A[:,7:11]).as_euler('xyz',degrees=True)[:,2]).max())
 metrics['boost_acceleration_m_s2']=BOOST_ACCEL
 metrics['landing_x_target_m']=float(recovery['x0']+recovery['vx']*LANDING_DURATION)
 metrics['final_descent_start_s']=descent_start
 active=(A[:,0]<ROLL_START)|(A[:,0]>=ROLL_END)
 metrics['position_tracking_RMSE_PID_active_m']=float(np.sqrt(np.mean(np.sum((A[active,1:4]-A[active,24:27])**2,axis=1))))
 metrics['landing_vertical_reference_duration_s']=LANDING_DURATION
 metrics['minimum_forward_speed_after_roll_m_s']=float(A[A[:,0]>=ROLL_END,4].min())
 metrics['landing_design']='C2 quintic vertical arc with constant forward speed; no hover, stop or vertical-only segment'
 metrics['final_descent_pitch_request_limit_deg']=None
 angles=Rotation.from_quat(A[:,7:11]).as_euler('xyz',degrees=True)
 near=(A[:,0]>ROLL_END)&(A[:,3]<2.)
 metrics['maximum_absolute_pitch_below_2m_deg']=float(abs(angles[near,1]).max()) if near.any() else None
 metrics['roll_completed']=bool(abs(metrics['net_roll_degrees']-360)<5)
 metrics['success']=metrics['success'] and metrics['roll_completed']
 return A,metrics

def animate(A):
 from render_3d import render
 render(A)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--no-animation',action='store_true');args=parser.parse_args()
 A,metrics=run();_,fine=run(.00125);_,wind=run(disturbance=True)
 metrics['half_dt_check']={'success':fine['success'],'clearance_difference_m':fine['minimum_conservative_vertical_clearance_m']-metrics['minimum_conservative_vertical_clearance_m'],'touchdown_vertical_speed_difference_m_s':fine['touchdown']['vertical_speed_m_s']-metrics['touchdown']['vertical_speed_m_s'] if fine['touchdown'] and metrics['touchdown'] else None}
 metrics['yaw_and_crosswind_check']=wind
 metrics['display_direction']='right to left in 3D; positive world x projects left'
 metrics['controller']='hybrid boost/recovery position PID; roll attitude PD with angular-rate/acceleration feedforward'
 metrics['roll_timing_s']=[ROLL_START,ROLL_END];metrics['recovery_duration_s']=RECOVERY_DURATION
 metrics['horizontal_braking_reference_duration_s']=None
 metrics['rendering']='3D with live telemetry HUD; nozzle hardware visible even when idle'
 metrics['dimensions_including_nozzles_m']=[4.5,2.3,1.2]
 metrics['assumptions']='2000 kg constant mass; inertia diag(900,3600,4000) kg m2; no calibrated Tesla parameters' 
 roll_angle=np.rad2deg(np.unwrap(Rotation.from_quat(A[:,7:11]).as_euler('xyz')[:,0]))
 angles_deg=Rotation.from_quat(A[:,7:11]).as_euler('xyz',degrees=True)
 telemetry=np.column_stack([A[:,0],A[:,3],A[:,3]/.3048,np.linalg.norm(A[:,14:24]@DIR,axis=1)/(MASS*GRAVITY),roll_angle,angles_deg[:,1:3],A[:,18:20],np.linalg.norm(A[:,4:7],axis=1)*3.6,A[:,6]])
 desired_angles=Rotation.from_quat(A[:,30:34]).as_euler('xyz',degrees=True)
 desired_roll=np.rad2deg(np.unwrap(np.deg2rad(desired_angles[:,0])))
 position_target=A[:,24:27].copy();position_target[(A[:,0]>=ROLL_START)&(A[:,0]<ROLL_END)]=np.nan
 phases=np.where(A[:,0]<ROLL_START,0,np.where(A[:,0]<ROLL_END,1,np.where(A[:,0]<metrics['final_descent_start_s'],2,3)))
 comparison=np.column_stack([A[:,0],phases,A[:,1:4],position_target,roll_angle,desired_roll,angles_deg[:,1],desired_angles[:,1],angles_deg[:,2],desired_angles[:,2],A[:,14:20]])
 np.savetxt(ROOT/'control_comparison.csv',comparison,delimiter=',',header='time_s,phase_0boost_1roll_3smooth_landing,x_m,y_m,z_m,x_target_m,y_target_m,z_target_m,roll_deg,roll_target_deg,pitch_deg,pitch_target_deg,yaw_deg,yaw_target_deg,underside_front_left_N,underside_front_right_N,underside_rear_left_N,underside_rear_right_N,side_front_left_N,side_front_right_N',comments='')
 np.savetxt(ROOT/'telemetry.csv',telemetry,delimiter=',',header='time_s,com_altitude_m,com_altitude_ft,thrust_specific_force_g,unwrapped_roll_deg,pitch_deg,yaw_deg,front_left_actual_N,front_right_actual_N,speed_km_h,vertical_speed_m_s',comments='')
 np.savetxt(ROOT/'roll_history.csv',np.column_stack([A[:,0],roll_angle,np.rad2deg(A[:,11])]),delimiter=',',header='time_s,unwrapped_roll_deg,body_x_angular_velocity_deg_s',comments='')
 np.savetxt(ROOT/'trajectory.csv',A,delimiter=',',header='t,x,y,z,vx,vy,vz,qx,qy,qz,qw,wx,wy,wz,f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,xref,yref,zref,vxref,vyref,vzref,qd_x,qd_y,qd_z,qd_w',comments='')
 (ROOT/'metrics.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2))
 if not args.no_animation:animate(A)
