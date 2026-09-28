#include "gtcrn.h"
#include "common.h"
#include <map>
#include <string>
#include <cstdio>
int main(int argc, char** argv) {
    if (argc != 4) { fprintf(stderr, "usage: dump <gguf> <fixdir> <outdir>\n"); return 2; }
    GtcrnModel m;
    if (!m.load(argv[1])) { fprintf(stderr, "load fail\n"); return 1; }
    std::string fix = argv[2], out = argv[3];
    NpyArray e = npy_load(fix + "/in_spec_e.npy");
    NpyArray y = npy_load(fix + "/in_spec_y.npy");
    int T = (int)e.shape[2];
    std::map<std::string, NpyArray> cap;
    auto o = m.forward(e.data.data(), y.data.data(), T, &cap);
    for (auto& kv : cap) npy_save(out + "/" + kv.first + ".npy", kv.second);
    NpyArray os; os.shape = {1, 257, T, 2}; os.data = o;
    npy_save(out + "/out_spec.npy", os);
    printf("dumped %zu stages to %s\n", cap.size(), out.c_str());
    return 0;
}
