#include <bits/stdc++.h>
#include <parallel/algorithm>
#include <omp.h>
using namespace std;

struct Rec { uint64_t code; uint64_t cfg; };
struct CmpRec { bool operator()(Rec const&a, Rec const&b) const { if(a.code!=b.code) return a.code<b.code; return a.cfg<b.cfg; } };

static inline uint8_t getn(uint64_t c,int i){return (c>>(4*i))&15ULL;}
static inline uint64_t setn(uint64_t c,int i,uint8_t v){return c | (uint64_t(v)<<(4*i));}

uint64_t make_D(int mode,int a,int b){
    // mode: 0 C,c ; 1 S,c ; 2 C,s ; 3 S,s
    bool activeS = mode&1;
    bool passiveS = mode&2;
    uint64_t code=0;
    for(int x=0;x<16;x++){
        int p = passiveS ? max(0,a-x) : (x<=a?a:0);
        int A = activeS ? max(0,x-b) : (x>=b?x:0);
        int y=max(p,A);
        code=setn(code,x,y);
    }
    return code;
}
uint64_t make_T(int mode,int a,int b,bool full){
    // canonical modes 0:C,s,c 1:S,s,c 2:C,c,s 3:S,c,s
    // full modes p,q,A encoded: pS bit2, qS bit1, activeS bit0 (0..7)
    bool pS,qS,activeS;
    if(full){ pS=(mode>>2)&1; qS=(mode>>1)&1; activeS=mode&1; }
    else {
        if(mode<2){pS=true;qS=false;activeS=mode&1;}
        else {pS=false;qS=true;activeS=mode&1;}
    }
    uint64_t code=0;
    for(int x=0;x<16;x++){
        int p = pS ? max(0,a-x) : (x<=a?a:0);
        int q = qS ? max(0,b-x) : (x<=b?b:0);
        int y = activeS ? max(0,p-q) : (p>=q?p:0);
        code=setn(code,x,y);
    }
    return code;
}

vector<Rec> base(char kind,bool full){
    vector<Rec> v;
    int modes = kind=='D'?4:(full?8:4);
    v.reserve(modes*256);
    for(int m=0;m<modes;m++) for(int a=0;a<16;a++) for(int b=0;b<16;b++){
        uint64_t code= kind=='D'?make_D(m,a,b):make_T(m,a,b,full);
        uint64_t cfg=(uint64_t)(m*256+a*16+b);
        v.push_back({code,cfg});
    }
    __gnu_parallel::sort(v.begin(),v.end(),CmpRec());
    vector<Rec> u; u.reserve(v.size());
    for(auto &r:v) if(u.empty()||u.back().code!=r.code) u.push_back(r);
    return u;
}

void make_lut(uint64_t outer, vector<uint16_t>& lut){
    lut.resize(65536);
    uint8_t mapv[16]; for(int i=0;i<16;i++) mapv[i]=getn(outer,i);
    for(int z=0;z<65536;z++){
        uint16_t o=0;
        o |= uint16_t(mapv[(z>>0)&15])<<0;
        o |= uint16_t(mapv[(z>>4)&15])<<4;
        o |= uint16_t(mapv[(z>>8)&15])<<8;
        o |= uint16_t(mapv[(z>>12)&15])<<12;
        lut[z]=o;
    }
}
static inline uint64_t apply_lut(uint64_t x, vector<uint16_t> const& lut){
    uint64_t y=0;
    y |= uint64_t(lut[(x>>0)&0xFFFFULL])<<0;
    y |= uint64_t(lut[(x>>16)&0xFFFFULL])<<16;
    y |= uint64_t(lut[(x>>32)&0xFFFFULL])<<32;
    y |= uint64_t(lut[(x>>48)&0xFFFFULL])<<48;
    return y;
}

vector<Rec> compose(vector<Rec> const& inner, vector<Rec> const& outer, int depth){
    size_t N=inner.size(), M=outer.size();
    unsigned long long total=(unsigned long long)N*M;
    cerr<<"candidates "<<total<<"\n";
    vector<Rec> all; all.resize(total);
    #pragma omp parallel
    {
        vector<uint16_t> lut(65536);
        #pragma omp for schedule(dynamic,1)
        for(long long j=0;j<(long long)M;j++){
            make_lut(outer[j].code,lut);
            size_t off=(size_t)j*N;
            for(size_t i=0;i<N;i++){
                uint64_t code=apply_lut(inner[i].code,lut);
                uint64_t cfg=inner[i].cfg | (outer[j].cfg << (11*(depth-1)));
                all[off+i]={code,cfg};
            }
        }
    }
    cerr<<"sorting...\n";
    __gnu_parallel::sort(all.begin(),all.end(),CmpRec());
    cerr<<"unique...\n";
    vector<Rec> u; u.reserve(min<size_t>(all.size(),10000000));
    for(auto &r:all) if(u.empty()||u.back().code!=r.code) u.push_back(r);
    all.clear(); all.shrink_to_fit();
    return u;
}

int rev(uint16_t mask){ int r=0; int prev=mask&1; for(int i=1;i<16;i++){int b=(mask>>i)&1; r+=b!=prev; prev=b;} return r; }

struct BRep { bool seen=false; uint64_t analog=0,cfg=0; int k=0,op=0; };
void metrics(vector<Rec> const& v, string const& tag, bool writeBin){
    static int totalByR[16]; for(int r=0;r<16;r++) totalByR[r]=2* (int)round(tgamma(16)/(tgamma(r+1)*tgamma(16-r))); // 2*C15r
    vector<BRep> br(65536);
    for(auto &r:v){
        uint8_t vals[16]; for(int i=0;i<16;i++) vals[i]=getn(r.code,i);
        for(int k=0;k<16;k++){
            uint16_t mask=0; for(int i=0;i<16;i++) if(vals[i]>k) mask|=(1u<<i);
            uint16_t masks[2]={mask,(uint16_t)(mask^0xFFFFu)};
            for(int op=0;op<2;op++) if(!br[masks[op]].seen){br[masks[op]]={true,r.code,r.cfg,k,op};}
        }
    }
    int cnt=0,dist[16]={0};
    for(int m=0;m<65536;m++) if(br[m].seen){cnt++;dist[rev((uint16_t)m)]++;}
    int kfull=-1; for(int r=0;r<16;r++){ if(dist[r]==totalByR[r])kfull=r; else break; }
    cout<<tag<<" analog="<<v.size()<<" binary="<<cnt<<" kfull="<<kfull<<" rev:";
    for(int r=0;r<16;r++) if(dist[r]) cout<<" "<<r<<":"<<dist[r];
    cout<<"\n";
    if(writeBin){
        ofstream f(tag+"_binary.tsv");
        f<<"mask\tanalog_hex\tcfgpack\tk\top\trev\n";
        for(int m=0;m<65536;m++) if(br[m].seen){
            f<<m<<"\t"<<hex<<setw(16)<<setfill('0')<<br[m].analog<<dec<<"\t"<<br[m].cfg<<"\t"<<br[m].k<<"\t"<<(br[m].op?"le":"gt")<<"\t"<<rev((uint16_t)m)<<"\n";
        }
    }
}

void writeAnalog(vector<Rec> const& v,string const& fn){
    ofstream f(fn,ios::binary); uint64_t n=v.size(); f.write((char*)&n,8); f.write((char*)v.data(),v.size()*sizeof(Rec));
}
vector<Rec> readAnalog(string const& fn){
    ifstream f(fn,ios::binary); uint64_t n;f.read((char*)&n,8); vector<Rec> v(n);f.read((char*)v.data(),n*sizeof(Rec));return v;
}

int main(int argc,char**argv){
    if(argc<4){cerr<<"usage: dt_enum canonical|full SEQ OUTPREFIX [innerfile]\n";return 1;}
    bool full=string(argv[1])=="full"; string seq=argv[2], pref=argv[3];
    vector<Rec> cur=base(seq[0],full);
    if(seq.size()==1){ metrics(cur,pref,true); writeAnalog(cur,pref+"_analog.bin"); return 0; }
    // Optional load precomputed inner of length seq-1 to speed depth3
    if(argc>=5){ cur=readAnalog(argv[4]); }
    else {
        for(size_t pos=1;pos<seq.size();pos++){
            auto out=base(seq[pos],full);
            cur=compose(cur,out,pos+1);
        }
    }
    if(argc>=5){ auto out=base(seq.back(),full); cur=compose(cur,out,seq.size()); }
    metrics(cur,pref,true); writeAnalog(cur,pref+"_analog.bin");
    return 0;
}
