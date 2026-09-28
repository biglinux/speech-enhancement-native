// Times a SPA audio.aec plugin: RTF = wall / audio. Usage:
//   run_aec_timed <so> <rate> <mic.f32> <ref.f32> <model|-> <repeats>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <spa/support/plugin.h>
#include <spa/utils/dict.h>
#include <spa/param/audio/raw.h>
#include <spa/interfaces/audio/aec.h>
typedef int (*enum_fn)(const struct spa_handle_factory **, uint32_t *);
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }
static float* readf(const char*p, size_t*n){ FILE*f=fopen(p,"rb"); if(!f)return NULL; fseek(f,0,SEEK_END); long b=ftell(f); fseek(f,0,SEEK_SET); *n=b/4; float*x=malloc(b); fread(x,4,*n,f); fclose(f); return x; }
int main(int argc,char**argv){
  if(argc!=7){fprintf(stderr,"usage: %s so rate mic ref model|- repeats\n",argv[0]);return 2;}
  const char*so=argv[1]; uint32_t rate=atoi(argv[2]); const char*model=argv[5]; int reps=atoi(argv[6]);
  void*dl=dlopen(so,RTLD_NOW); if(!dl){fprintf(stderr,"dlopen %s\n",dlerror());return 1;}
  enum_fn efn=(enum_fn)dlsym(dl,"spa_handle_factory_enum");
  const struct spa_handle_factory*factory=NULL,*f; uint32_t idx=0;
  while(efn(&f,&idx)==1) if(f&&f->name&&!strcmp(f->name,"audio.aec")){factory=f;break;}
  if(!factory){fprintf(stderr,"no audio.aec\n");return 1;}
  struct spa_handle*h=calloc(1,spa_handle_factory_get_size(factory,NULL));
  spa_handle_factory_init(factory,h,NULL,NULL,0);
  struct spa_audio_aec*aec=NULL; spa_handle_get_interface(h,SPA_TYPE_INTERFACE_AUDIO_AEC,(void**)&aec);
  struct spa_dict_item items[1]; int ni=0;
  if(strcmp(model,"-")) items[ni++]=(struct spa_dict_item){"gtcrn.model",model};
  struct spa_dict args=SPA_DICT_INIT(items,ni);
  struct spa_audio_info_raw info={0}; info.format=0x206; /* F32P planar */ info.rate=rate; info.channels=1;
  int r=spa_audio_aec_init2(aec,&args,&info,&info,&info); if(r<0) r=spa_audio_aec_init(aec,&args,&info);
  if(r<0){fprintf(stderr,"init %d\n",r);return 1;}
  spa_audio_aec_activate(aec);
  size_t nm,nr; float*mic=readf(argv[3],&nm),*ref=readf(argv[4],&nr); size_t N=nm<nr?nm:nr;
  uint32_t fr=rate/100; float*out=malloc(fr*4);
  const float*rc[1],*pc[1]; float*oc[1]; oc[0]=out;
  double wall=0; size_t total=0;
  for(int rep=0;rep<reps;rep++){
    for(size_t o=0;o+fr<=N;o+=fr){ rc[0]=mic+o; pc[0]=ref+o;
      double t0=now(); spa_audio_aec_run(aec,rc,pc,oc,fr); wall+=now()-t0; total+=fr; }
  }
  double audio=(double)total/rate;
  printf("AEC audio=%.1fs wall=%.2fs RTF=%.3f\n",audio,wall,wall/audio);
  return 0;
}
