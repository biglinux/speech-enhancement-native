// Times a LADSPA plugin: RTF = wall / audio. Usage:
//   run_ladspa_timed <so> <label> <rate> <frames> <repeats> [controls...]
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ladspa.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(int argc,char**argv){
  if(argc<6){fprintf(stderr,"usage: %s so label rate frames repeats [ctl...]\n",argv[0]);return 2;}
  const char*so=argv[1],*label=argv[2]; unsigned long rate=atol(argv[3]); unsigned long fr=atol(argv[4]); int reps=atoi(argv[5]);
  void*dl=dlopen(so,RTLD_NOW); if(!dl){fprintf(stderr,"dlopen %s\n",dlerror());return 1;}
  LADSPA_Descriptor_Function df=(LADSPA_Descriptor_Function)dlsym(dl,"ladspa_descriptor");
  const LADSPA_Descriptor*d=NULL; for(unsigned long i=0;(d=df(i));i++) if(!strcmp(d->Label,label)) break;
  if(!d){fprintf(stderr,"label %s not found\n",label);return 1;}
  LADSPA_Handle h=d->instantiate(d,rate);
  float*in=calloc(fr,4),*out=calloc(fr,4); int ci=6;
  float ctl[32]; int nc=0;
  for(unsigned long p=0;p<d->PortCount;p++){
    if(LADSPA_IS_PORT_AUDIO(d->PortDescriptors[p])){
      if(LADSPA_IS_PORT_INPUT(d->PortDescriptors[p])) d->connect_port(h,p,in); else d->connect_port(h,p,out);
    } else { // control
      float v = ci<argc? atof(argv[ci++]) : 0.0f; ctl[nc]=v; d->connect_port(h,p,&ctl[nc]); nc++;
    }
  }
  if(d->activate) d->activate(h);
  // fill input with low-level noise so the model actually works
  for(unsigned long i=0;i<fr;i++) in[i]=0.01f*((float)(i%97)/97.0f-0.5f);
  double wall=0; unsigned long total=0;
  for(int r=0;r<reps;r++){ double t0=now(); d->run(h,fr); wall+=now()-t0; total+=fr; }
  double audio=(double)total/rate;
  printf("LADSPA[%s] audio=%.1fs wall=%.2fs RTF=%.3f\n",label,audio,wall,wall/audio);
  return 0;
}
