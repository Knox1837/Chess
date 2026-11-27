// engine.cpp - fast alpha-beta chess engine (UCI) with hand-written and NNUE evaluation.
#include "chess.hpp"

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

using namespace chess;
using Clock = std::chrono::steady_clock;

namespace {

// All output goes through here: the search thread and the command loop both print, and an
// unsynchronised std::cout can duplicate or garble lines.
std::mutex outMutex;
void emit(const std::string& line) {
    std::lock_guard<std::mutex> g(outMutex);
    std::cout << line << '\n';
    std::cout.flush();
}

constexpr int INF = 1000000, MATE = 100000, MAX_PLY = 64, MAX_CP = 2500;
constexpr int MG_VALUE[7] = {0, 82, 337, 365, 477, 1025, 0};
constexpr int EG_VALUE[7] = {0, 94, 281, 297, 512, 936, 0};
constexpr int PHASE_INC[7] = {0, 0, 1, 1, 2, 4, 0};
constexpr int ORDER_VAL[7] = {0, 100, 320, 330, 500, 900, 20000};
constexpr int PASSED_MG[8] = {0, 0, 5, 10, 20, 35, 60, 0};
constexpr int PASSED_EG[8] = {0, 5, 15, 30, 55, 90, 140, 0};

// generated from the Python tables (index 0 = a8, White's view)
static const int MG_PST[7][64] = {
    {0},
    {0,0,0,0,0,0,0,0,98,134,61,95,68,126,34,-11,-6,7,26,31,65,56,25,-20,-14,13,6,21,23,12,17,-23,-27,-2,-5,12,17,6,10,-25,-26,-4,-4,-10,3,3,33,-12,-35,-1,-20,-23,-15,24,38,-22,0,0,0,0,0,0,0,0},
    {-167,-89,-34,-49,61,-97,-15,-107,-73,-41,72,36,23,62,7,-17,-47,60,37,65,84,129,73,44,-9,17,19,53,37,69,18,22,-13,4,16,13,28,19,21,-8,-23,-9,12,10,19,17,25,-16,-29,-53,-12,-3,-1,18,-14,-19,-105,-21,-58,-33,-17,-28,-19,-23},
    {-29,4,-82,-37,-25,-42,7,-8,-26,16,-18,-13,30,59,18,-47,-16,37,43,40,35,50,37,-2,-4,5,19,50,37,37,7,-2,-6,13,13,26,34,12,10,4,0,15,15,15,14,27,18,10,4,15,16,0,7,21,33,1,-33,-3,-14,-21,-13,-12,-39,-21},
    {32,42,32,51,63,9,31,43,27,32,58,62,80,67,26,44,-5,19,26,36,17,45,61,16,-24,-11,7,26,24,35,-8,-20,-36,-26,-12,-1,9,-7,6,-23,-45,-25,-16,-17,3,0,-5,-33,-44,-16,-20,-9,-1,11,-6,-71,-19,-13,1,17,16,7,-37,-26},
    {-28,0,29,12,59,44,43,45,-24,-39,-5,1,-16,57,28,54,-13,-17,7,8,29,56,47,57,-27,-27,-16,-16,-1,17,-2,1,-9,-26,-9,-10,-2,-4,3,-3,-14,2,-11,-2,-5,2,14,5,-35,-8,11,2,8,15,-3,1,-1,-18,-9,10,-15,-25,-31,-50},
    {-65,23,16,-15,-56,-34,2,13,29,-1,-20,-7,-8,-4,-38,-29,-9,24,2,-16,-20,6,22,-22,-17,-20,-12,-27,-30,-25,-14,-36,-49,-1,-27,-39,-46,-44,-33,-51,-14,-14,-22,-46,-44,-30,-15,-27,1,7,-8,-64,-43,-16,9,8,-15,36,12,-54,8,-28,24,14}
};
static const int EG_PST[7][64] = {
    {0},
    {0,0,0,0,0,0,0,0,178,173,158,134,147,132,165,187,94,100,85,67,56,53,82,84,32,24,13,5,-2,4,17,17,13,9,-3,-7,-7,-8,3,-1,4,7,-6,1,0,-5,-1,-8,13,8,8,10,13,0,2,-7,0,0,0,0,0,0,0,0},
    {-58,-38,-13,-28,-31,-27,-63,-99,-25,-8,-25,-2,-9,-25,-24,-52,-24,-20,10,9,-1,-9,-19,-41,-17,3,22,22,22,11,8,-18,-18,-6,16,25,16,17,4,-18,-23,-3,-1,15,10,-3,-20,-22,-42,-20,-10,-5,-2,-20,-23,-44,-29,-51,-23,-15,-22,-18,-50,-64},
    {-14,-21,-11,-8,-7,-9,-17,-24,-8,-4,7,-12,-3,-13,-4,-14,2,-8,0,-1,-2,6,0,4,-3,9,12,9,14,10,3,2,-6,3,13,19,7,10,-3,-9,-12,-3,8,10,13,3,-7,-15,-14,-18,-7,-1,4,-9,-15,-27,-23,-9,-23,-5,-9,-16,-5,-17},
    {13,10,18,15,12,12,8,5,11,13,13,11,-3,3,8,3,7,7,7,5,4,-3,-5,-3,4,3,13,1,2,1,-1,2,3,5,8,4,-5,-6,-8,-11,-4,0,-5,-1,-7,-12,-8,-16,-6,-6,0,2,-9,-9,-11,-3,-9,2,3,-1,-5,-13,4,-20},
    {-9,22,22,27,27,19,10,20,-17,20,32,41,58,25,30,0,-20,6,9,49,47,35,19,9,3,22,24,45,57,40,57,36,-18,28,19,47,31,34,39,23,-16,-27,15,6,9,17,10,5,-22,-23,-30,-16,-16,-23,-36,-32,-33,-28,-22,-43,-5,-32,-20,-41},
    {-74,-35,-18,-18,-11,15,4,-17,-12,17,14,17,17,38,23,11,10,17,23,15,20,45,44,13,-8,22,24,27,26,33,26,3,-18,-4,21,24,27,23,9,-11,-19,-3,11,21,23,16,7,-9,-27,-11,4,13,14,4,-5,-17,-53,-34,-21,-11,-28,-14,-24,-43}
};

#ifdef _MSC_VER
inline int popcnt(uint64_t b) { return int(__popcnt64(b)); }
inline int ctz(uint64_t b) { unsigned long i; _BitScanForward64(&i, b); return int(i); }
#else
inline int popcnt(uint64_t b) { return __builtin_popcountll(b); }
inline int ctz(uint64_t b) { return __builtin_ctzll(b); }
#endif

const PieceType PTS[6] = {PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP,
                          PieceType::ROOK, PieceType::QUEEN,  PieceType::KING};

//  Hand-written evaluation (identical to the Python engine's evaluate())
int MGT[2][7][64], EGT[2][7][64];
uint64_t FILE_BB[8], ADJ_FILES[8], PASSED_MASK[2][64], FRONT_SAME[2][64], SHIELD[2][64];

uint64_t ranksAhead(int rank, bool white) {
    uint64_t m = 0;
    if (white) { for (int r = rank + 1; r < 8; ++r) m |= 0xFFULL << (8 * r); }
    else       { for (int r = 0; r < rank; ++r)     m |= 0xFFULL << (8 * r); }
    return m;
}

// Evaluation weights. The defaults reproduce the original hand-written evaluation exactly; tune_eval.py can
// fit new ones and the EvalParams option loads them from a text file. [0] = middlegame, [1] = endgame.
struct EvalParams {
    int pst[2][7][64];            // piece value + piece-square table, indexed like MG_PST (a8 first, White's view)
    int passed[2][8];             // passed pawn bonus by relative rank
    int iso[2], dbl[2], rookOpen[2], rookSemi[2], bpair[2], shield[2];
    int tempo;
    int mob[2][4], kz[2][4];      // knight, bishop, rook, queen: mobility area / attacks on the enemy king zone
    bool dynamic;                 // any mobility or king-zone weight is non-zero (needs attack maps)
};
EvalParams EP;
EvalParams BUILTIN;      // the weights used when EvalParams is empty: the baked tuned weights if any, else the originals

// Tuned weights written by `python tune_eval.py bake` (empty = keep the original hand-written weights)
#include "eval_default.inc"

void finalizeParams(EvalParams& p) {
    p.dynamic = false;
    for (int ph = 0; ph < 2; ++ph)
        for (int t = 0; t < 4; ++t)
            if (p.mob[ph][t] || p.kz[ph][t]) p.dynamic = true;
}

void setDefaultParams(EvalParams& p) {
    std::memset(&p, 0, sizeof(p));
    for (int pt = 1; pt <= 6; ++pt)
        for (int i = 0; i < 64; ++i) {
            p.pst[0][pt][i] = MG_VALUE[pt] + MG_PST[pt][i];
            p.pst[1][pt][i] = EG_VALUE[pt] + EG_PST[pt][i];
        }
    for (int r = 0; r < 8; ++r) { p.passed[0][r] = PASSED_MG[r]; p.passed[1][r] = PASSED_EG[r]; }
    p.iso[0] = -10;      p.iso[1] = -15;
    p.dbl[0] = -12;      p.dbl[1] = -20;
    p.rookOpen[0] = 20;  p.rookOpen[1] = 10;
    p.rookSemi[0] = 10;  p.rookSemi[1] = 5;
    p.bpair[0] = 30;     p.bpair[1] = 50;
    p.shield[0] = 9;     p.shield[1] = 0;
    p.tempo = 10;
    finalizeParams(p);
}

void rebuildTables() {
    for (int pt = 1; pt <= 6; ++pt)
        for (int sq = 0; sq < 64; ++sq) {
            MGT[0][pt][sq] = EP.pst[0][pt][sq ^ 56];
            EGT[0][pt][sq] = EP.pst[1][pt][sq ^ 56];
            MGT[1][pt][sq] = EP.pst[0][pt][sq];
            EGT[1][pt][sq] = EP.pst[1][pt][sq];
        }
}

std::string dumpParams(const EvalParams& p) {
    std::ostringstream o;
    o << "version 1\n";
    for (int ph = 0; ph < 2; ++ph)
        for (int pt = 1; pt <= 6; ++pt) {
            o << (ph ? "pst_eg " : "pst_mg ") << pt;
            for (int i = 0; i < 64; ++i) o << ' ' << p.pst[ph][pt][i];
            o << '\n';
        }
    auto line = [&](const char* name, const int* v, int n) {
        o << name;
        for (int i = 0; i < n; ++i) o << ' ' << v[i];
        o << '\n';
    };
    line("passed_mg", p.passed[0], 8);  line("passed_eg", p.passed[1], 8);
    line("isolated", p.iso, 2);         line("doubled", p.dbl, 2);
    line("rook_open", p.rookOpen, 2);   line("rook_semi", p.rookSemi, 2);
    line("bishop_pair", p.bpair, 2);    line("shield", p.shield, 2);
    line("tempo", &p.tempo, 1);
    line("mob_mg", p.mob[0], 4);        line("mob_eg", p.mob[1], 4);
    line("kz_mg", p.kz[0], 4);          line("kz_eg", p.kz[1], 4);
    return o.str();
}

// Parses a parameter file (format of dumpParams) on top of the values already in p. Unknown keys are ignored.
bool parseParams(std::istream& f, EvalParams& p, std::string& err) {
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream is(line);
        std::string key;
        if (!(is >> key) || key[0] == '#') continue;
        auto rd = [&](int* dst, int n) {
            for (int i = 0; i < n; ++i)
                if (!(is >> dst[i])) return false;
            return true;
        };
        bool ok = true;
        if (key == "pst_mg" || key == "pst_eg") {
            int pt = 0;
            if (!(is >> pt) || pt < 1 || pt > 6) ok = false;
            else ok = rd(p.pst[key == "pst_eg" ? 1 : 0][pt], 64);
        }
        else if (key == "passed_mg") ok = rd(p.passed[0], 8);
        else if (key == "passed_eg") ok = rd(p.passed[1], 8);
        else if (key == "isolated") ok = rd(p.iso, 2);
        else if (key == "doubled") ok = rd(p.dbl, 2);
        else if (key == "rook_open") ok = rd(p.rookOpen, 2);
        else if (key == "rook_semi") ok = rd(p.rookSemi, 2);
        else if (key == "bishop_pair") ok = rd(p.bpair, 2);
        else if (key == "shield") ok = rd(p.shield, 2);
        else if (key == "tempo") ok = rd(&p.tempo, 1);
        else if (key == "mob_mg") ok = rd(p.mob[0], 4);
        else if (key == "mob_eg") ok = rd(p.mob[1], 4);
        else if (key == "kz_mg") ok = rd(p.kz[0], 4);
        else if (key == "kz_eg") ok = rd(p.kz[1], 4);
        if (!ok) { err = "bad line: " + key; return false; }
    }
    finalizeParams(p);
    return true;
}

bool loadParams(const std::string& path, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "cannot open file"; return false; }
    EvalParams p;
    setDefaultParams(p);
    if (!parseParams(f, p, err)) return false;
    EP = p;
    rebuildTables();
    return true;
}

void initEval() {
    for (int f = 0; f < 8; ++f) FILE_BB[f] = 0x0101010101010101ULL << f;
    for (int f = 0; f < 8; ++f)
        ADJ_FILES[f] = (f > 0 ? FILE_BB[f - 1] : 0) | (f < 7 ? FILE_BB[f + 1] : 0);
    setDefaultParams(BUILTIN);
    if (BAKED_EVAL_PARAMS[0]) {
        std::istringstream baked(BAKED_EVAL_PARAMS);
        std::string err;
        EvalParams p = BUILTIN;
        if (parseParams(baked, p, err)) BUILTIN = p;
    }
    EP = BUILTIN;
    rebuildTables();
    for (int sq = 0; sq < 64; ++sq) {
        int f = sq & 7, r = sq >> 3;
        for (int ci = 0; ci < 2; ++ci) {
            bool white = ci == 0;
            uint64_t ahead = ranksAhead(r, white);
            PASSED_MASK[ci][sq] = (FILE_BB[f] | ADJ_FILES[f]) & ahead;
            FRONT_SAME[ci][sq] = FILE_BB[f] & ahead;
            int step = white ? 1 : -1;
            uint64_t m = 0;
            for (int dr = 1; dr <= 2; ++dr) {
                int rr = r + step * dr;
                if (rr < 0 || rr > 7) continue;
                for (int ff = f - 1; ff <= f + 1; ++ff)
                    if (ff >= 0 && ff < 8) m |= 1ULL << (rr * 8 + ff);
            }
            SHIELD[ci][sq] = m;
        }
    }
}

int evalHand(const Board& bd) {
    int mg[2] = {0, 0}, eg[2] = {0, 0}, phase = 0;
    uint64_t pawns[2] = {bd.pieces(PieceType::PAWN, Color::WHITE).getBits(),
                         bd.pieces(PieceType::PAWN, Color::BLACK).getBits()};
    for (int ci = 0; ci < 2; ++ci) {
        Color color = ci == 0 ? Color::WHITE : Color::BLACK;
        int m = 0, e = 0;
        uint64_t ownp = pawns[ci], oppp = pawns[1 - ci];
        for (int pt = 1; pt <= 6; ++pt) {
            uint64_t bb = bd.pieces(PTS[pt - 1], color).getBits();
            if (!bb) continue;
            const int* tm = MGT[ci][pt];
            const int* te = EGT[ci][pt];
            uint64_t it = bb;
            while (it) {
                int sq = ctz(it);
                it &= it - 1;
                m += tm[sq];
                e += te[sq];
                phase += PHASE_INC[pt];
                if (pt == 1) {
                    int r = ci == 0 ? (sq >> 3) : 7 - (sq >> 3);
                    int f = sq & 7;
                    if (!(oppp & PASSED_MASK[ci][sq])) { m += EP.passed[0][r]; e += EP.passed[1][r]; }
                    if (!(ownp & ADJ_FILES[f])) { m += EP.iso[0]; e += EP.iso[1]; }
                    if (ownp & FRONT_SAME[ci][sq]) { m += EP.dbl[0]; e += EP.dbl[1]; }
                } else if (pt == 4) {
                    uint64_t fm = FILE_BB[sq & 7];
                    if (!(ownp & fm)) {
                        if (!(oppp & fm)) { m += EP.rookOpen[0]; e += EP.rookOpen[1]; }
                        else              { m += EP.rookSemi[0]; e += EP.rookSemi[1]; }
                    }
                }
            }
            if (pt == 3 && popcnt(bb) >= 2) { m += EP.bpair[0]; e += EP.bpair[1]; }
            else if (pt == 6) {
                const int cnt = popcnt(ownp & SHIELD[ci][ctz(bb)]);
                m += EP.shield[0] * cnt;
                e += EP.shield[1] * cnt;
            }
        }
        mg[ci] = m;
        eg[ci] = e;
    }
    if (EP.dynamic) {
        // Mobility and king-zone terms from the union of squares each piece type attacks
        const uint64_t occ = bd.occ().getBits();
        const Bitboard occBB(occ);
        uint64_t att[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}}, zone[2] = {0, 0}, kingbb[2] = {0, 0}, area[2], pa[2];
        pa[0] = ((pawns[0] << 9) & ~FILE_BB[0]) | ((pawns[0] << 7) & ~FILE_BB[7]);
        pa[1] = ((pawns[1] >> 7) & ~FILE_BB[0]) | ((pawns[1] >> 9) & ~FILE_BB[7]);
        for (int ci = 0; ci < 2; ++ci) {
            const Color color = ci == 0 ? Color::WHITE : Color::BLACK;
            kingbb[ci] = bd.pieces(PieceType::KING, color).getBits();
            for (int t = 0; t < 4; ++t) {
                uint64_t bb = bd.pieces(PTS[t + 1], color).getBits();
                uint64_t a = 0;
                while (bb) {
                    const Square sq(ctz(bb));
                    bb &= bb - 1;
                    a |= t == 0 ? attacks::knight(sq).getBits()
                       : t == 1 ? attacks::bishop(sq, occBB).getBits()
                       : t == 2 ? attacks::rook(sq, occBB).getBits()
                                : attacks::queen(sq, occBB).getBits();
                }
                att[ci][t] = a;
            }
            if (kingbb[ci]) zone[ci] = attacks::king(Square(ctz(kingbb[ci]))).getBits() | kingbb[ci];
        }
        for (int ci = 0; ci < 2; ++ci) area[ci] = ~(pawns[ci] | kingbb[ci]) & ~pa[1 - ci];
        for (int ci = 0; ci < 2; ++ci)
            for (int t = 0; t < 4; ++t) {
                const int mob = popcnt(att[ci][t] & area[ci]);
                const int kza = popcnt(att[ci][t] & zone[1 - ci]);
                mg[ci] += EP.mob[0][t] * mob + EP.kz[0][t] * kza;
                eg[ci] += EP.mob[1][t] * mob + EP.kz[1][t] * kza;
            }
    }
    if (phase > 24) phase = 24;
    int mgs = mg[0] - mg[1], egs = eg[0] - eg[1];
    int score = (mgs * phase + egs * (24 - phase)) / 24;   // truncates toward 0, like the Python int()
    return bd.sideToMove() == Color::WHITE ? score + EP.tempo : -score + EP.tempo;
}

//  NNUE (v2: king-bucketed, both perspectives) - weights from nnue.bin written by Python
//  File: "NNU2" | int32 H | W1[6144*H] b1[H] W2[2H*32] b2[32] W3[32*32] b3[32] W4[32] b4  (float32)
struct Net {
    int H = 0;
    std::vector<float> W1, b1, W2, b2, W3, b3, W4;
    float b4 = 0;
    bool ok = false;
    // Quantized copy used by the fast search path (first layer x255, second layer x64, int16)
    std::vector<int16_t> qW1, qb1, qW2t;   // qW2t is transposed: [32][2H]
} net;

constexpr float QA = 255.0f, QB = 64.0f;

void quantizeNet() {
    auto q = [](float v, float scale) {
        const float r = std::round(v * scale);
        return int16_t(std::max(-32767.0f, std::min(32767.0f, r)));
    };
    const int H = net.H;
    net.qW1.resize(net.W1.size());
    for (size_t i = 0; i < net.W1.size(); ++i) net.qW1[i] = q(net.W1[i], QA);
    net.qb1.resize(H);
    for (int i = 0; i < H; ++i) net.qb1[i] = q(net.b1[i], QA);
    net.qW2t.resize(size_t(32) * 2 * H);
    for (int i = 0; i < 2 * H; ++i)
        for (int j = 0; j < 32; ++j) net.qW2t[size_t(j) * 2 * H + i] = q(net.W2[size_t(i) * 32 + j], QB);
}

bool loadNet(const std::string& path) {
    net = Net();
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[4];
    int32_t H = 0;
    f.read(magic, 4);
    f.read(reinterpret_cast<char*>(&H), 4);
    if (!f || std::memcmp(magic, "NNU2", 4) != 0 || H <= 0 || H > 1024) return false;
    auto rd = [&](std::vector<float>& v, size_t n) {
        v.resize(n);
        f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float));
    };
    net.H = H;
    rd(net.W1, size_t(6144) * H); rd(net.b1, H);
    rd(net.W2, size_t(2) * H * 32); rd(net.b2, 32);
    rd(net.W3, 32 * 32); rd(net.b3, 32);
    rd(net.W4, 32);
    f.read(reinterpret_cast<char*>(&net.b4), 4);
    net.ok = bool(f);
    if (net.ok) quantizeNet();
    return net.ok;
}

// Perspective c: 0 = White, 1 = Black. Each perspective has its own accumulator (king bucket of that colour's king).
inline int kingBucket(const Board& bd, int c) {
    const int ks = bd.kingSq(c == 0 ? Color::WHITE : Color::BLACK).index() ^ (c == 0 ? 0 : 56);
    return ((ks & 7) >> 1) + ((ks >> 3) >= 2 ? 4 : 0);
}

// First layer for perspective c rebuilt from scratch (before the clipped ReLU)
void netFullAcc(const Board& bd, int c, int bucket, float* a) {
    const int H = net.H;
    const int flip = c == 0 ? 0 : 56;
    std::memcpy(a, net.b1.data(), H * sizeof(float));
    for (int cc = 0; cc < 2; ++cc) {
        const Color col = cc == 0 ? Color::WHITE : Color::BLACK;
        for (int t = 0; t < 6; ++t) {
            uint64_t bits = bd.pieces(PTS[t], col).getBits();
            const int base = bucket * 768 + (t + (cc == c ? 0 : 6)) * 64;
            while (bits) {
                const int sq = ctz(bits);
                bits &= bits - 1;
                const float* row = &net.W1[size_t(base + (sq ^ flip)) * H];
                for (int i = 0; i < H; ++i) a[i] += row[i];
            }
        }
    }
}

int netTail(float* h1);

// Clipped ReLU on both accumulators (a0 = side to move), then the dense layers
int netDense(const float* a0, const float* a1) {
    const int H = net.H;
    float h1[32];
    std::memcpy(h1, net.b2.data(), sizeof(h1));
    for (int side = 0; side < 2; ++side) {
        const float* a = side == 0 ? a0 : a1;
        for (int i = 0; i < H; ++i) {
            const float v = std::min(1.0f, std::max(0.0f, a[i]));
            if (v == 0.0f) continue;
            const float* w = &net.W2[size_t(side * H + i) * 32];
            for (int j = 0; j < 32; ++j) h1[j] += v * w[j];
        }
    }
    return netTail(h1);
}

// Last two layers (32 -> 32 -> 1), shared by the float and the quantized path
int netTail(float* h1) {
    float h2[32];
    std::memcpy(h2, net.b3.data(), sizeof(h2));
    for (int i = 0; i < 32; ++i) {
        const float v = std::min(1.0f, std::max(0.0f, h1[i]));
        if (v == 0.0f) continue;
        const float* w = &net.W3[i * 32];
        for (int j = 0; j < 32; ++j) h2[j] += v * w[j];
    }
    float out = net.b4;
    for (int j = 0; j < 32; ++j) out += std::min(1.0f, std::max(0.0f, h2[j])) * net.W4[j];
    int cp = int(out * 400.0f);
    return cp > MAX_CP ? MAX_CP : (cp < -MAX_CP ? -MAX_CP : cp);
}

// Reference evaluation: rebuilds everything for this position (used by the 'eval' command and NNUECheck)
int evalNet(const Board& bd) {
    float acc[2][1024];
    const int stmIdx = bd.sideToMove() == Color::WHITE ? 0 : 1;
    netFullAcc(bd, stmIdx, kingBucket(bd, stmIdx), acc[0]);
    netFullAcc(bd, 1 - stmIdx, kingBucket(bd, 1 - stmIdx), acc[1]);
    return netDense(acc[0], acc[1]);
}

// ---- Quantized fast path: int16 accumulators, int16 x int16 -> int32 dot products (AVX2 when available) ----
inline void addRow(int16_t* d, const int16_t* r, int H) {
    int i = 0;
#if defined(__AVX2__)
    for (; i + 16 <= H; i += 16)
        _mm256_storeu_si256((__m256i*)(d + i), _mm256_add_epi16(_mm256_loadu_si256((const __m256i*)(d + i)),
                                                                _mm256_loadu_si256((const __m256i*)(r + i))));
#endif
    for (; i < H; ++i) d[i] = int16_t(d[i] + r[i]);
}

inline void subRow(int16_t* d, const int16_t* r, int H) {
    int i = 0;
#if defined(__AVX2__)
    for (; i + 16 <= H; i += 16)
        _mm256_storeu_si256((__m256i*)(d + i), _mm256_sub_epi16(_mm256_loadu_si256((const __m256i*)(d + i)),
                                                                _mm256_loadu_si256((const __m256i*)(r + i))));
#endif
    for (; i < H; ++i) d[i] = int16_t(d[i] - r[i]);
}

void netFullAccQ(const Board& bd, int c, int bucket, int16_t* a) {
    const int H = net.H;
    const int flip = c == 0 ? 0 : 56;
    std::memcpy(a, net.qb1.data(), H * sizeof(int16_t));
    for (int cc = 0; cc < 2; ++cc) {
        const Color col = cc == 0 ? Color::WHITE : Color::BLACK;
        for (int t = 0; t < 6; ++t) {
            uint64_t bits = bd.pieces(PTS[t], col).getBits();
            const int base = bucket * 768 + (t + (cc == c ? 0 : 6)) * 64;
            while (bits) {
                const int sq = ctz(bits);
                bits &= bits - 1;
                addRow(a, &net.qW1[size_t(base + (sq ^ flip)) * H], H);
            }
        }
    }
}

inline int dotQ(const int16_t* x, const int16_t* w, int n) {
    int i = 0;
    int sum = 0;
#if defined(__AVX2__)
    __m256i acc = _mm256_setzero_si256();
    for (; i + 16 <= n; i += 16)
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_loadu_si256((const __m256i*)(x + i)),
                                                      _mm256_loadu_si256((const __m256i*)(w + i))));
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    sum = _mm_cvtsi128_si32(s);
#endif
    for (; i < n; ++i) sum += int(x[i]) * int(w[i]);
    return sum;
}

int netDenseQ(const int16_t* a0, const int16_t* a1) {
    const int H = net.H;
    alignas(32) int16_t act[2 * 1024];
    for (int i = 0; i < H; ++i) {
        act[i] = std::min<int16_t>(255, std::max<int16_t>(0, a0[i]));
        act[H + i] = std::min<int16_t>(255, std::max<int16_t>(0, a1[i]));
    }
    float h1[32];
    for (int j = 0; j < 32; ++j)
        h1[j] = net.b2[j] + float(dotQ(act, &net.qW2t[size_t(j) * 2 * H], 2 * H)) * (1.0f / (QA * QB));
    return netTail(h1);
}

// Quantized evaluation with a full rebuild (reference for the incremental path: results must be identical)
int evalNetQ(const Board& bd) {
    int16_t acc[2][1024];
    const int stmIdx = bd.sideToMove() == Color::WHITE ? 0 : 1;
    netFullAccQ(bd, stmIdx, kingBucket(bd, stmIdx), acc[0]);
    netFullAccQ(bd, 1 - stmIdx, kingBucket(bd, 1 - stmIdx), acc[1]);
    return netDenseQ(acc[0], acc[1]);
}

// Options
struct Options {
    int hashMB = 64;
    bool useNNUE = true;
    int contHist = 0;       // countermove move + 1-ply continuation history for quiet move ordering (0 = off)
    bool nnueCheck = false;   // debug: compare the incremental NNUE value with a full rebuild at every evaluation
    int noise = 0;
    int softPercent = 75;   // don't start a new search depth after this % of the move time
    int rfpMargin = 80;     // reverse futility pruning: cp per remaining depth (0 = off)
    int fpMargin = 120;     // futility pruning of quiet moves: cp per remaining depth (0 = off)
    int lmp = 1;            // late move pruning of quiet moves at low depth (0 = off)
    int lmr = 2;            // late move reductions: 0 = old fixed 1/2 ply, 1 = log table, 2 = log table + improving/killer/history/PV adjustments
    int histMalus = 1;      // bounded history with a penalty for quiet moves that failed to cut (0 = old additive history)
    int histLmr = 0;        // reduce moves with a good history score less, poor ones more (0 = off)
    int nullEval = 0;       // only try the null move when the static eval is at or above beta (0 = off)
    int nullDepth = 0;      // null-move reduction 3 + depth/4 instead of 3 (+1 above depth 6) (0 = off)
    int aspiration = 25;    // root aspiration window half-width in cp from depth 5 (0 = full window)
    int threads = 1;        // search threads (Lazy SMP: helpers share the transposition table)
    int timeManage = 1;     // with a game clock: spend more time when the best move or score is unstable, less when stable
    int timeMax = 300;      // hard limit for one move as % of the planned time (time management only)
    int moveOverhead = 30;  // ms kept in reserve per move for communication delay
    int see = 2;            // static exchange evaluation: 1 = order captures by it, 2 = also skip losing captures in quiescence
} opts;

// Late move reduction table: grows with depth and move number
int LMR_TABLE[64][64];
void initLMR() {
    for (int d = 1; d < 64; ++d)
        for (int m = 1; m < 64; ++m)
            LMR_TABLE[d][m] = int(0.75 + std::log(double(d)) * std::log(double(m)) / 2.25);
}
constexpr int HIST_MAX = 16384;
inline void updateHistory(int& h, int bonus) { h += bonus - h * std::abs(bonus) / HIST_MAX; }

inline int evaluate(const Board& bd) {
    return (opts.useNNUE && net.ok) ? evalNet(bd) : evalHand(bd);
}

// Transposition table, shared by all search threads. Each entry is two atomic words; the key is stored xor-ed with the data, so an entry torn by two threads writing at once simply fails the key check and is ignored.
struct TTEntry {
    std::atomic<uint64_t> key{0};
    std::atomic<uint64_t> data{0};
};
struct TTData { int score, depth, flag; uint16_t move; };
inline uint64_t ttPack(int score, int depth, int flag, uint16_t move) {
    return uint64_t(move) | (uint64_t(uint16_t(int16_t(score))) << 16) | (uint64_t(uint16_t(int16_t(depth))) << 32) |
           (uint64_t(flag) << 48);
}
inline TTData ttUnpack(uint64_t d) {
    return {int16_t(uint16_t(d >> 16)), int16_t(uint16_t(d >> 32)), int((d >> 48) & 3), uint16_t(d)};
}
std::unique_ptr<TTEntry[]> TT;
size_t TT_MASK = 0, TT_COUNT = 0;

void resizeTT(int mb) {
    size_t n = 1;
    while (n * 2 * sizeof(TTEntry) <= size_t(mb) * 1024 * 1024) n *= 2;
    TT.reset(new TTEntry[n]);
    TT_COUNT = n;
    TT_MASK = n - 1;
}
void clearTT() {
    std::memset(static_cast<void*>(TT.get()), 0, TT_COUNT * sizeof(TTEntry));   // lock-free atomics are plain words
}
inline int toTT(int s, int ply) { return s > MATE - 1000 ? s + ply : (s < -MATE + 1000 ? s - ply : s); }
inline int fromTT(int s, int ply) { return s > MATE - 1000 ? s - ply : (s < -MATE + 1000 ? s + ply : s); }

// Search
struct Scored { Move m; int s; };

std::atomic<uint64_t> nnueChecked{0}, nnueMismatch{0}, nnueDiffSum{0};
std::atomic<int> nnueDiffMax{0};

class Search {
   public:
    Board bd;
    uint64_t nodes = 0;
    std::atomic<bool>* stopFlag = nullptr;
    bool stopped = false;
    Clock::time_point deadline;
    bool timed = false;
    Move killers[MAX_PLY + 2][2];
    int history[2][64][64];
    // Countermove + continuation history, indexed by the previous move's (piece, to-square) slot = piece * 64 + to
    static constexpr int SLOTS = 12 * 64;
    std::vector<int16_t> contHist;   // [SLOTS][SLOTS]
    Move counterMv[SLOTS];
    int prevSlot[MAX_PLY + 4];       // slot of the move that led to each ply (-1 at the root and after a null move)
    int slotOf(const Move& m) const { return int(bd.at(m.from())) * 64 + m.to().index(); }
    static constexpr int NO_EVAL = -2 * INF;
    int evalStack[MAX_PLY + 4];   // static eval per ply (NO_EVAL when unknown or in check), for the 'improving' flag
    std::mt19937_64 rng{std::random_device{}()};

    // NNUE accumulators: one entry per ply, filled lazily. Each perspective (0 = White, 1 = Black) is updated from the
    // nearest computed ancestor by adding/removing the few features that changed, or rebuilt when its king changed bucket.
    struct AccInfo {
        int nchg;
        int16_t chg[4];          // plane * 64 + square (plane = colour * 6 + piece type)
        int8_t sign[4];          // +1 piece appeared, -1 piece disappeared
        int bucket[2];
        bool refresh[2];         // king bucket changed on the move into this ply: needs a rebuild
        bool computed[2];
    };
    AccInfo accInfo[MAX_PLY + 4];
    std::vector<int16_t> accBuf; // [(MAX_PLY + 4)][2][H], quantized
    bool nnueActive = false;

    int16_t* accPtr(int ply, int c) { return &accBuf[(size_t(ply) * 2 + c) * net.H]; }

    void initAcc() {
        nnueActive = opts.useNNUE && net.ok;
        if (!nnueActive) return;
        const size_t need = size_t(MAX_PLY + 4) * 2 * net.H;
        if (accBuf.size() != need) accBuf.assign(need, int16_t(0));
        AccInfo& a = accInfo[0];
        a.nchg = 0;
        for (int c = 0; c < 2; ++c) {
            a.bucket[c] = kingBucket(bd, c);
            a.refresh[c] = true;
            a.computed[c] = false;
        }
    }

    void doMove(const Move& m, int ply) {
        if (opts.contHist) prevSlot[ply + 1] = slotOf(m);
        if (!nnueActive) { bd.makeMove(m); return; }
        uint64_t pre[12];
        for (int cc = 0; cc < 2; ++cc)
            for (int t = 0; t < 6; ++t)
                pre[cc * 6 + t] = bd.pieces(PTS[t], cc == 0 ? Color::WHITE : Color::BLACK).getBits();
        bd.makeMove(m);
        AccInfo& a = accInfo[ply + 1];
        const AccInfo& p = accInfo[ply];
        a.nchg = 0;
        for (int c = 0; c < 2; ++c) { a.bucket[c] = p.bucket[c]; a.refresh[c] = false; a.computed[c] = false; }
        bool kingMoved[2] = {false, false};
        bool overflow = false;
        for (int cc = 0; cc < 2; ++cc)
            for (int t = 0; t < 6; ++t) {
                const int plane = cc * 6 + t;
                const uint64_t post = bd.pieces(PTS[t], cc == 0 ? Color::WHITE : Color::BLACK).getBits();
                uint64_t diff = pre[plane] ^ post;
                if (!diff) continue;
                if (t == 5) kingMoved[cc] = true;
                while (diff) {
                    const int sq = ctz(diff);
                    diff &= diff - 1;
                    if (a.nchg < 4) {
                        a.chg[a.nchg] = int16_t(plane * 64 + sq);
                        a.sign[a.nchg] = (post >> sq) & 1 ? 1 : -1;
                        ++a.nchg;
                    } else overflow = true;
                }
            }
        for (int c = 0; c < 2; ++c) {
            if (kingMoved[c]) {
                const int nb = kingBucket(bd, c);
                if (nb != p.bucket[c]) { a.refresh[c] = true; a.bucket[c] = nb; }
            }
            if (overflow) a.refresh[c] = true;
        }
    }

    void doNull(int ply) {
        prevSlot[ply + 1] = -1;
        bd.makeNullMove();
        if (!nnueActive) return;
        AccInfo& a = accInfo[ply + 1];
        const AccInfo& p = accInfo[ply];
        a.nchg = 0;
        for (int c = 0; c < 2; ++c) { a.bucket[c] = p.bucket[c]; a.refresh[c] = false; a.computed[c] = false; }
    }

    void updateAcc(int c, int n) {
        int j = n;
        for (;; --j) {
            if (accInfo[j].computed[c]) break;
            if (accInfo[j].refresh[c]) { j = -1; break; }   // entry 0 always has refresh set, so this ends
        }
        if (j < 0) {
            netFullAccQ(bd, c, accInfo[n].bucket[c], accPtr(n, c));
            accInfo[n].computed[c] = true;
            return;
        }
        const int H = net.H;
        const int flip = c == 0 ? 0 : 56;
        for (int k = j + 1; k <= n; ++k) {
            int16_t* dst = accPtr(k, c);
            std::memcpy(dst, accPtr(k - 1, c), H * sizeof(int16_t));
            const AccInfo& a = accInfo[k];
            for (int e = 0; e < a.nchg; ++e) {
                const int plane = a.chg[e] >> 6, sq = a.chg[e] & 63;
                const int cc = plane >= 6 ? 1 : 0, t = plane - cc * 6;
                const int16_t* row = &net.qW1[size_t(a.bucket[c] * 768 + (t + (cc == c ? 0 : 6)) * 64 + (sq ^ flip)) * H];
                if (a.sign[e] > 0) addRow(dst, row, H);
                else subRow(dst, row, H);
            }
            accInfo[k].computed[c] = true;
        }
    }

    int evalPos(int ply) {
        if (!nnueActive) return evalHand(bd);
        updateAcc(0, ply);
        updateAcc(1, ply);
        const int stmIdx = bd.sideToMove() == Color::WHITE ? 0 : 1;
        const int v = netDenseQ(accPtr(ply, stmIdx), accPtr(ply, 1 - stmIdx));
        if (opts.nnueCheck) {
            ++nnueChecked;
            if (v != evalNetQ(bd)) ++nnueMismatch;            // incremental vs rebuild: integer math, must be identical
            const int d = std::abs(v - evalNet(bd));          // quantized vs float: the accuracy cost of int16
            nnueDiffSum += d;
            int m = nnueDiffMax.load();
            while (d > m && !nnueDiffMax.compare_exchange_weak(m, d)) {}
        }
        return v;
    }

    void reset(bool hasTime, double seconds) {
        initAcc();
        nodes = 0;
        stopped = false;
        timed = hasTime;
        if (hasTime) deadline = Clock::now() + std::chrono::microseconds(int64_t(seconds * 1e6));
        std::memset(history, 0, sizeof(history));
        if (opts.contHist) {
            contHist.assign(size_t(SLOTS) * SLOTS, int16_t(0));
            for (Move& c : counterMv) c = Move();
        }
        for (int& v : prevSlot) v = -1;
        for (int& v : evalStack) v = NO_EVAL;
        for (auto& k : killers) k[0] = k[1] = Move();
    }

    bool timeUp() {
        if (stopFlag && stopFlag->load(std::memory_order_relaxed)) return true;
        return timed && Clock::now() >= deadline;
    }

    // Static exchange evaluation: true if capturing on m.to() and then recapturing with the least valuable
    // attacker each time leaves the mover with at least `threshold` centipawns. Pins are ignored.
    bool seeGe(const Move& m, int threshold) const {
        if (m.typeOf() == Move::CASTLING) return threshold <= 0;
        const Square to = m.to();
        const Square from = m.from();
        const bool ep = m.typeOf() == Move::ENPASSANT;
        const int victim = ep ? 1 : int(bd.at<PieceType>(to)) + 1;       // index into ORDER_VAL (0 = empty)
        int swap = ORDER_VAL[victim] - threshold;
        if (swap < 0) return false;
        swap = ORDER_VAL[int(bd.at<PieceType>(from)) + 1] - swap;
        if (swap <= 0) return true;

        uint64_t occ = bd.occ().getBits() ^ (1ULL << from.index()) ^ (1ULL << to.index());
        Color stm = bd.sideToMove();
        if (ep) occ ^= 1ULL << (to.index() + (stm == Color::WHITE ? -8 : 8));   // captured pawn leaves the board

        const uint64_t pawns = bd.pieces(PieceType::PAWN).getBits();
        const uint64_t knights = bd.pieces(PieceType::KNIGHT).getBits();
        const uint64_t bishops = bd.pieces(PieceType::BISHOP).getBits();
        const uint64_t rooks = bd.pieces(PieceType::ROOK).getBits();
        const uint64_t queens = bd.pieces(PieceType::QUEEN).getBits();
        const uint64_t kings = bd.pieces(PieceType::KING).getBits();
        const uint64_t diag = bishops | queens;
        const uint64_t line = rooks | queens;
        const uint64_t white = bd.us(Color::WHITE).getBits();
        const uint64_t black = bd.us(Color::BLACK).getBits();

        uint64_t attackers = (attacks::pawn(Color::BLACK, to).getBits() & pawns & white) |
                             (attacks::pawn(Color::WHITE, to).getBits() & pawns & black) |
                             (attacks::knight(to).getBits() & knights) | (attacks::king(to).getBits() & kings) |
                             (attacks::bishop(to, Bitboard(occ)).getBits() & diag) |
                             (attacks::rook(to, Bitboard(occ)).getBits() & line);
        int res = 1;
        while (true) {
            stm = ~stm;
            attackers &= occ;
            const uint64_t mine = attackers & (stm == Color::WHITE ? white : black);
            if (!mine) break;
            res ^= 1;
            uint64_t bb;
            if ((bb = mine & pawns)) {
                if ((swap = ORDER_VAL[1] - swap) < res) break;
                occ ^= bb & (~bb + 1);
                attackers |= attacks::bishop(to, Bitboard(occ)).getBits() & diag;
            } else if ((bb = mine & knights)) {
                if ((swap = ORDER_VAL[2] - swap) < res) break;
                occ ^= bb & (~bb + 1);
            } else if ((bb = mine & bishops)) {
                if ((swap = ORDER_VAL[3] - swap) < res) break;
                occ ^= bb & (~bb + 1);
                attackers |= attacks::bishop(to, Bitboard(occ)).getBits() & diag;
            } else if ((bb = mine & rooks)) {
                if ((swap = ORDER_VAL[4] - swap) < res) break;
                occ ^= bb & (~bb + 1);
                attackers |= attacks::rook(to, Bitboard(occ)).getBits() & line;
            } else if ((bb = mine & queens)) {
                if ((swap = ORDER_VAL[5] - swap) < res) break;
                occ ^= bb & (~bb + 1);
                attackers |= (attacks::bishop(to, Bitboard(occ)).getBits() & diag) |
                             (attacks::rook(to, Bitboard(occ)).getBits() & line);
            } else {   // king: only captures if the other side has no attacker left
                return (attackers & ~(stm == Color::WHITE ? white : black)) ? bool(res ^ 1) : bool(res);
            }
        }
        return bool(res);
    }

    int orderMoves(const Movelist& ml, Scored* out, Move ttMove, int ply) {
        const int stm = bd.sideToMove() == Color::WHITE ? 0 : 1;
        int n = 0;
        for (const Move& m : ml) {
            int s;
            if (m.move() == ttMove.move() && ttMove.move() != 0) s = 10000000;
            else if (bd.isCapture(m)) {
                int victim = m.typeOf() == Move::ENPASSANT ? 1 : int(bd.at<PieceType>(m.to())) + 1;
                int attacker = int(bd.at<PieceType>(m.from())) + 1;
                s = 1000000 + 10 * ORDER_VAL[victim] - ORDER_VAL[attacker] / 10;
                if (m.typeOf() == Move::PROMOTION) s += 5000;
                else if (opts.see && !seeGe(m, 0)) s -= 2000000;   // losing capture: after the quiet moves
            } else if (m.typeOf() == Move::PROMOTION) s = 900000 + int(m.promotionType());
            else if (m == killers[ply][0]) s = 800000;
            else if (m == killers[ply][1]) s = 790000;
            else if (opts.contHist) {
                const int prev = prevSlot[ply];
                if (prev >= 0 && counterMv[prev] == m) s = 780000;      // the reply that refuted the previous move before
                else {
                    s = history[stm][m.from().index()][m.to().index()];
                    if (prev >= 0) s += contHist[size_t(prev) * SLOTS + slotOf(m)];
                }
            } else s = history[stm][m.from().index()][m.to().index()];
            out[n++] = {m, s};
        }
        return n;
    }

    static void pickBest(Scored* a, int n, int i) {
        int b = i;
        for (int j = i + 1; j < n; ++j) if (a[j].s > a[b].s) b = j;
        if (b != i) std::swap(a[i], a[b]);
    }

    int qsearch(int alpha, int beta, int ply, int qd) {
        ++nodes;
        if ((nodes & 2047) == 0 && timeUp()) stopped = true;
        if (stopped) return 0;
        if (ply >= MAX_PLY) return evalPos(ply);

        const bool inCheck = bd.inCheck();
        Movelist ml;
        int stand;
        if (inCheck && qd < 3) {
            movegen::legalmoves(ml, bd);
            if (ml.empty()) return -MATE + ply;
            stand = -INF;
        } else {
            stand = evalPos(ply);
            if (stand >= beta) return stand;
            if (stand > alpha) alpha = stand;
            if (stand + 1000 < alpha) return stand;
            movegen::legalmoves<movegen::MoveGenType::CAPTURE>(ml, bd);
        }
        Scored mv[256];
        const int n = orderMoves(ml, mv, Move(), ply);
        int best = stand;
        for (int i = 0; i < n; ++i) {
            pickBest(mv, n, i);
            const Move m = mv[i].m;
            if (!inCheck) {
                int victim = m.typeOf() == Move::ENPASSANT ? 1 : int(bd.at<PieceType>(m.to())) + 1;
                if (stand + ORDER_VAL[victim] + 200 < alpha && m.typeOf() != Move::PROMOTION) continue;
                if (opts.see >= 2 && m.typeOf() != Move::PROMOTION && !seeGe(m, 0)) continue;   // losing capture
            }
            doMove(m, ply);
            const int score = -qsearch(-beta, -alpha, ply + 1, qd + 1);
            bd.unmakeMove(m);
            if (stopped) return 0;
            if (score > best) {
                best = score;
                if (score > alpha) {
                    alpha = score;
                    if (alpha >= beta) break;
                }
            }
        }
        return best;
    }

    int negamax(int depth, int alpha, int beta, int ply, bool allowNull) {
        ++nodes;
        if ((nodes & 2047) == 0 && timeUp()) stopped = true;
        if (stopped) return 0;

        if (ply > 0) {
            if (bd.isHalfMoveDraw()) return 0;
            if (bd.isRepetition(1)) return 0;
            if (bd.occ().count() <= 4 && bd.isInsufficientMaterial()) return 0;
        }
        if (ply >= MAX_PLY) return evalPos(ply);

        const bool inCheck = bd.inCheck();
        if (inCheck) ++depth;
        if (depth <= 0) return qsearch(alpha, beta, ply, 0);

        alpha = std::max(alpha, -MATE + ply);
        beta = std::min(beta, MATE - ply - 1);
        if (alpha >= beta) return alpha;

        const int origAlpha = alpha;
        const uint64_t key = bd.hash();
        TTEntry& e = TT[key & TT_MASK];
        Move ttMove{Move::NO_MOVE};
        const uint64_t eKey = e.key.load(std::memory_order_relaxed);
        const uint64_t eData = e.data.load(std::memory_order_relaxed);
        const TTData tt = ttUnpack(eData);
        const bool ttHit = (eKey ^ eData) == key;
        if (ttHit) {
            ttMove = Move(tt.move);
            if (tt.depth >= depth) {
                const int s = fromTT(tt.score, ply);
                if (tt.flag == 0) return s;
                if (tt.flag == 1 && s > alpha) alpha = s;
                else if (tt.flag == 2 && s < beta) beta = s;
                if (alpha >= beta) return s;
            }
        }

        // Static evaluation, only computed where a pruning rule can use it
        const bool pvNode = beta - alpha > 1;
        const bool nonPawn = bd.hasNonPawnMaterial(bd.sideToMove());
        int staticEval = 0;
        const bool useEval = !pvNode && !inCheck && depth <= 6 && (opts.rfpMargin > 0 || opts.fpMargin > 0);
        bool haveEval = useEval;
        if (useEval) staticEval = evalPos(ply);

        // 'Improving': static eval better than two plies ago (LMR=2 only)
        evalStack[ply] = NO_EVAL;
        bool improving = true;
        if (opts.lmr >= 2 && !inCheck && depth >= 3) {
            if (!haveEval) { staticEval = evalPos(ply); haveEval = true; }
            evalStack[ply] = staticEval;
            if (ply >= 2 && evalStack[ply - 2] != NO_EVAL) improving = staticEval > evalStack[ply - 2];
        }

        // Reverse futility pruning: far above beta at low depth, so the node would fail high anyway
        if (useEval && opts.rfpMargin > 0 && nonPawn && std::abs(beta) < MATE - 1000 &&
            staticEval - opts.rfpMargin * depth >= beta)
            return staticEval - opts.rfpMargin * depth;

        if (allowNull && !inCheck && depth >= 3 && beta < MATE - 1000 && bd.hasNonPawnMaterial(bd.sideToMove())) {
            if (opts.nullEval && !haveEval) { staticEval = evalPos(ply); haveEval = true; }
            if (!opts.nullEval || staticEval >= beta) {
                const int R = opts.nullDepth ? 3 + depth / 4 : 3 + (depth > 6);
                doNull(ply);
                const int s = -negamax(depth - R, -beta, -beta + 1, ply + 1, false);
                bd.unmakeNullMove();
                if (stopped) return 0;
                if (s >= beta) return beta;
            }
        }

        Movelist ml;
        movegen::legalmoves(ml, bd);
        if (ml.empty()) return inCheck ? -MATE + ply : 0;
        Scored mv[256];
        const int n = orderMoves(ml, mv, ttMove, ply);
        const int stm = bd.sideToMove() == Color::WHITE ? 0 : 1;

        int best = -INF;
        Move bestMove{Move::NO_MOVE};
        Move quiets[64];       // quiet moves tried so far at this node (for the history penalty)
        int nq = 0;
        for (int i = 0; i < n; ++i) {
            pickBest(mv, n, i);
            const Move m = mv[i].m;
            const bool quiet = !bd.isCapture(m) && m.typeOf() != Move::PROMOTION;
            const int mslot = opts.contHist ? slotOf(m) : 0;   // before the move is made
            doMove(m, ply);
            // Futility and late-move pruning of quiet moves (never in check, never when giving check)
            if (i > 0 && quiet && !pvNode && !inCheck && best > -MATE + 1000 && !bd.inCheck()) {
                if (opts.fpMargin > 0 && depth <= 3 && staticEval + opts.fpMargin * depth <= alpha) {
                    bd.unmakeMove(m);
                    continue;
                }
                if (opts.lmp && depth <= 3 && i >= 3 + depth * depth) {
                    bd.unmakeMove(m);
                    continue;
                }
            }
            int score;
            if (i == 0) {
                score = -negamax(depth - 1, -beta, -alpha, ply + 1, true);
            } else {
                int red = 0;
                if (opts.lmr) {
                    if (depth >= 3 && i >= 3 && quiet && !inCheck && !bd.inCheck()) {
                        red = LMR_TABLE[std::min(depth, 63)][std::min(i, 63)];
                        if (pvNode) --red;
                        if (opts.lmr >= 2) {
                            if (!improving) ++red;                                   // position is getting worse: reduce more
                            if (m == killers[ply][0] || m == killers[ply][1]) --red; // killers are likely good: reduce less
                            int hs = history[stm][m.from().index()][m.to().index()];
                            if (opts.contHist && prevSlot[ply] >= 0) hs += contHist[size_t(prevSlot[ply]) * SLOTS + mslot];
                            red -= std::max(-2, std::min(2, hs / 4096));
                        } else if (opts.histLmr)
                            red -= std::max(-2, std::min(2, history[stm][m.from().index()][m.to().index()] / 4096));
                        red = std::max(0, std::min(red, depth - 2));
                    }
                } else if (depth >= 3 && i >= 4 && quiet && !inCheck && !bd.inCheck())
                    red = (i >= 10 && depth >= 5) ? 2 : 1;
                score = -negamax(depth - 1 - red, -alpha - 1, -alpha, ply + 1, true);
                if (score > alpha && (red || score < beta))
                    score = -negamax(depth - 1, -beta, -alpha, ply + 1, true);
            }
            bd.unmakeMove(m);
            if (stopped) return 0;
            if (score > best) {
                best = score;
                bestMove = m;
                if (score > alpha) {
                    alpha = score;
                    if (alpha >= beta) {
                        if (quiet) {
                            if (!(killers[ply][0] == m)) { killers[ply][1] = killers[ply][0]; killers[ply][0] = m; }
                            int& h = history[stm][m.from().index()][m.to().index()];
                            if (opts.histMalus) {
                                const int bonus = std::min(300 * depth - 250, 2400);
                                updateHistory(h, bonus);
                                for (int j = 0; j < nq; ++j)       // earlier quiet moves did not cut: penalise them
                                    updateHistory(history[stm][quiets[j].from().index()][quiets[j].to().index()], -bonus);
                                if (opts.contHist && prevSlot[ply] >= 0) {
                                    const size_t row = size_t(prevSlot[ply]) * SLOTS;
                                    counterMv[prevSlot[ply]] = m;
                                    auto upd = [&](int slot, int b) {
                                        int v = contHist[row + slot];
                                        updateHistory(v, b);
                                        contHist[row + slot] = int16_t(v);
                                    };
                                    upd(mslot, bonus);
                                    for (int j = 0; j < nq; ++j) upd(slotOf(quiets[j]), -bonus);
                                }
                            } else {
                                h = std::min(h + depth * depth, 1000000);
                            }
                        }
                        break;
                    }
                }
            }
            if (quiet && nq < 64) quiets[nq++] = m;
        }

        const uint8_t flag = best <= origAlpha ? 2 : (best >= beta ? 1 : 0);
        if (!ttHit || depth >= tt.depth) {
            const uint64_t d = ttPack(toTT(best, ply), depth, flag, bestMove.move());
            e.data.store(d, std::memory_order_relaxed);
            e.key.store(key ^ d, std::memory_order_relaxed);
        }
        return best;
    }

    // Root drivers
    static void printInfo(int depth, int score, uint64_t nodes, double secs, const Move& pv) {
        std::ostringstream o;
        o << "info depth " << depth << " score ";
        if (std::abs(score) > MATE - 100) {
            int plies = MATE - std::abs(score);
            int moves = (plies + 1) / 2;
            o << "mate " << (score > 0 ? moves : -moves);
        } else o << "cp " << score;
        o << " nodes " << nodes << " nps " << uint64_t(nodes / std::max(secs, 1e-3))
          << " time " << int(secs * 1000);
        if (pv.move() != 0) o << " pv " << uci::moveToUci(pv);
        emit(o.str());
    }

    int helperIdx = 0;      // 0 = main thread; Lazy SMP helpers skip some depths and print nothing
    double optimum = 0;     // planned time for this move in seconds when time management is on (0 = fixed time rules)

    // Iterative deepening with optional aspiration windows. Don't start a new depth after softPercent of the time.
    Move iterative(const Movelist& root, int maxDepth, double seconds, bool hasTime, int& scoreOut) {
        static const int SKIP_SIZE[20] = {1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4};
        static const int SKIP_PHASE[20] = {0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7};
        reset(hasTime, seconds);
        const auto t0 = Clock::now();
        Move best = root[0];
        int bestScore = 0;
        int lastChange = 1;             // depth at which the best move last changed (time management)
        for (int depth = 1; depth <= maxDepth; ++depth) {
            const int prevScore = bestScore;
            if (helperIdx > 0 && depth > 1) {
                const int h = (helperIdx - 1) % 20;
                if (((depth + SKIP_PHASE[h]) / SKIP_SIZE[h]) % 2) continue;
            }
            int delta = opts.aspiration;
            int alpha0 = -INF, beta0 = INF;
            if (delta > 0 && depth >= 5 && std::abs(bestScore) < MATE - 1000) {
                alpha0 = bestScore - delta;
                beta0 = bestScore + delta;
            }
            Move itBest{Move::NO_MOVE};
            int itScore = -INF;
            while (true) {
                Scored mv[256];
                const int n = orderMoves(root, mv, best, 0);
                int alpha = alpha0;
                itBest = Move{Move::NO_MOVE};
                itScore = -INF;
                for (int i = 0; i < n; ++i) {
                    pickBest(mv, n, i);
                    const Move m = mv[i].m;
                    doMove(m, 0);
                    int s;
                    if (i == 0) s = -negamax(depth - 1, -beta0, -alpha, 1, true);
                    else {
                        s = -negamax(depth - 1, -alpha - 1, -alpha, 1, true);
                        if (s > alpha) s = -negamax(depth - 1, -beta0, -alpha, 1, true);
                    }
                    bd.unmakeMove(m);
                    if (stopped) break;
                    if (s > itScore) { itScore = s; itBest = m; }
                    if (s > alpha) alpha = s;
                    if (s >= beta0) break;                       // fail high: widen and search again
                }
                if (stopped) break;
                if (alpha0 > -INF && itScore <= alpha0) {        // fail low
                    beta0 = (alpha0 + beta0) / 2;
                    alpha0 = std::max(-INF, alpha0 - delta);
                    delta += delta / 2 + 5;
                } else if (beta0 < INF && itScore >= beta0) {    // fail high
                    beta0 = std::min(INF, beta0 + delta);
                    delta += delta / 2 + 5;
                } else break;
                if (delta > 600) { alpha0 = -INF; beta0 = INF; }
            }
            // a stopped re-search that had only failed low says nothing reliable about the best move
            const bool trust = !(stopped && alpha0 > -INF && itScore <= alpha0);
            if (itBest.move() != 0 && trust) {
                if (!(itBest == best)) lastChange = depth;
                best = itBest;
                bestScore = itScore;
            }
            const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
            if (helperIdx == 0 && (!stopped || itBest.move() != 0)) printInfo(depth, bestScore, nodes, secs, best);
            if (stopped || std::abs(bestScore) > MATE - 100) break;
            if (helperIdx == 0 && hasTime) {
                double plan = seconds;                  // fixed rule: the time limit itself
                if (optimum > 0) {
                    // Planned time, scaled by how settled the search looks. Between depths we stop once the elapsed time passes softPercent of it; a depth already running may continue to the hard limit.
                    double scale = 1.0;
                    if (depth >= 6) {
                        const int stable = depth - lastChange;
                        scale = stable >= 8 ? 0.55 : stable >= 5 ? 0.70 : stable >= 3 ? 0.85 : 1.0;
                        if (stable == 0) scale *= 1.25;                       // the best move just changed
                        const int drop = prevScore - bestScore;
                        if (drop > 15) scale *= 1.0 + std::min(0.6, (drop - 15) / 120.0);   // score is falling
                    }
                    plan = optimum * scale;
                }
                if (secs > plan * opts.softPercent / 100.0) break;
            }
        }
        scoreOut = bestScore;
        return best;
    }

    // Weak levels: score every move at a fixed depth, add noise, play the best noisy score.
    Move noisyRoot(const Movelist& root, int depth, int sigma, double seconds, bool hasTime, int& scoreOut) {
        reset(hasTime, seconds);
        Scored mv[256];
        const int n = orderMoves(root, mv, Move(), 0);
        std::normal_distribution<double> gauss(0.0, std::max(1, sigma));
        Move best = mv[0].m;
        double bestNoisy = -1e18;
        int bestTrue = 0;
        for (int i = 0; i < n; ++i) {
            doMove(mv[i].m, 0);
            const int s = -negamax(depth - 1, -INF, INF, 1, true);
            bd.unmakeMove(mv[i].m);
            if (stopped) break;
            const double noisy = s > -MATE + 1000 ? s + gauss(rng) : s - 10000.0;
            if (noisy > bestNoisy) { bestNoisy = noisy; best = mv[i].m; bestTrue = s; }
        }
        scoreOut = bestTrue;
        return best;
    }
};

//  UCI
Board gameBoard;
std::atomic<bool> stopFlag{false};
std::thread searchThread;
Search searcher;
std::atomic<bool> helperStop{false};
std::vector<std::unique_ptr<Search>> helpers;

void joinSearch() {
    if (searchThread.joinable()) searchThread.join();
}

void doGo(Board board, int depth, double movetime, bool hasTime, int noise, double optimum) {
    searcher.bd = board;
    searcher.stopFlag = &stopFlag;
    searcher.optimum = optimum;
    Movelist root;
    movegen::legalmoves(root, searcher.bd);
    if (root.empty()) { emit("bestmove 0000"); return; }
    int score = 0;
    Move best{Move::NO_MOVE};
    const auto t0 = Clock::now();
    if (root.size() == 1) {
        best = root[0];
    } else if (noise > 0) {
        best = searcher.noisyRoot(root, depth, noise, movetime, hasTime, score);
        const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
        Search::printInfo(depth, score, searcher.nodes, secs, best);
    } else {
        // Lazy SMP: helper threads search the same root with different depth patterns and share the table
        const int extra = opts.threads > 1 ? opts.threads - 1 : 0;
        std::vector<std::thread> pool;
        helperStop = false;
        while (int(helpers.size()) < extra) helpers.push_back(std::make_unique<Search>());
        for (int i = 0; i < extra; ++i) {
            Search* h = helpers[i].get();
            h->bd = searcher.bd;
            h->stopFlag = &helperStop;
            h->helperIdx = i + 1;
            pool.emplace_back([h, &root, depth, movetime, hasTime] {
                int sc = 0;
                h->iterative(root, depth, movetime, hasTime, sc);
            });
        }
        best = searcher.iterative(root, depth, movetime, hasTime, score);
        helperStop = true;
        for (auto& t : pool) t.join();
    }
    emit("bestmove " + uci::moveToUci(best));
}

void setPosition(std::istringstream& in) {
    std::string tok;
    in >> tok;
    Board b;
    if (tok == "startpos") b = Board();
    else if (tok == "fen") {
        std::string fen, part;
        while (in >> part && part != "moves") fen += part + " ";
        b = Board(fen);
        tok = part;
    }
    if (tok != "moves") in >> tok;
    if (tok == "moves") {
        std::string mv;
        while (in >> mv) b.makeMove(uci::uciToMove(b, mv));
    }
    gameBoard = b;
}

void bench() {
    const char* fens[] = {
        "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 b - - 0 7",
        "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 0 9",
        "2r2rk1/pp3ppp/2n1pn2/q1bp4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 w - - 0 11",
        "8/5pk1/6p1/8/3R4/6P1/5PK1/3r4 w - - 0 40"};
    uint64_t total = 0;
    const auto t0 = Clock::now();
    for (const char* f : fens) {
        Board b(f);
        searcher.bd = b;
        searcher.stopFlag = nullptr;
        Movelist root;
        movegen::legalmoves(root, searcher.bd);
        int sc;
        clearTT();
        searcher.iterative(root, 9, 0, false, sc);
        total += searcher.nodes;
    }
    const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
    if (opts.nnueCheck) {
        const uint64_t n = std::max<uint64_t>(1, nnueChecked.load());
        emit("nnuecheck evaluations " + std::to_string(nnueChecked.load()) + " mismatches " + std::to_string(nnueMismatch.load()) +
             " | int16 vs float: mean abs diff " + std::to_string(double(nnueDiffSum.load()) / double(n)) + " cp, max " + std::to_string(nnueDiffMax.load()) + " cp");
    }
    emit("bench nodes " + std::to_string(total) + " time " + std::to_string(int(secs * 1000)) +
         " nps " + std::to_string(uint64_t(total / secs)));
}

}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    initEval();
    initLMR();
    resizeTT(opts.hashMB);
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string cmd;
        in >> cmd;
        if (cmd == "uci") {
            emit("id name PyChessCpp 1.0\nid author you\n"
                 "option name Hash type spin default 64 min 1 max 4096\n"
                 "option name UseNNUE type check default true\n"
                 "option name ContHist type spin default 0 min 0 max 1\n"
                 "option name NNUECheck type check default false\n"
                 "option name EvalFile type string default <empty>\n"
                 "option name EvalParams type string default <empty>\n"
                 "option name Noise type spin default 0 min 0 max 1000\n"
                 "option name SoftTime type spin default 75 min 20 max 100\n"
                 "option name RFP type spin default 80 min 0 max 400\n"
                 "option name FP type spin default 120 min 0 max 600\n"
                 "option name LMP type spin default 1 min 0 max 1\n"
                 "option name LMR type spin default 2 min 0 max 2\n"
                 "option name HistMalus type spin default 1 min 0 max 1\n"
                 "option name HistLMR type spin default 0 min 0 max 1\n"
                 "option name SEE type spin default 2 min 0 max 2\n"
                 "option name NullEval type spin default 0 min 0 max 1\n"
                 "option name NullDepth type spin default 0 min 0 max 1\n"
                 "option name Aspiration type spin default 25 min 0 max 200\n"
                 "option name Threads type spin default 1 min 1 max 64\n"
                 "option name TimeManage type spin default 1 min 0 max 1\n"
                 "option name TimeMax type spin default 300 min 100 max 1000\n"
                 "option name MoveOverhead type spin default 30 min 0 max 5000\nuciok");
        } else if (cmd == "isready") {
            emit("readyok");
        } else if (cmd == "ucinewgame") {
            joinSearch();
            clearTT();
        } else if (cmd == "setoption") {
            std::string tok, name, value;
            in >> tok;  // "name"
            while (in >> tok && tok != "value") name += (name.empty() ? "" : " ") + tok;
            std::getline(in, value);
            while (!value.empty() && value[0] == ' ') value.erase(0, 1);
            if (name == "Hash") { joinSearch(); opts.hashMB = std::max(1, std::stoi(value)); resizeTT(opts.hashMB); }
            else if (name == "UseNNUE") opts.useNNUE = (value == "true" || value == "1");
            else if (name == "ContHist") opts.contHist = std::stoi(value) != 0;
            else if (name == "NNUECheck") opts.nnueCheck = (value == "true" || value == "1");
            else if (name == "Noise") opts.noise = std::max(0, std::stoi(value));
            else if (name == "SoftTime") opts.softPercent = std::min(100, std::max(20, std::stoi(value)));
            else if (name == "RFP") opts.rfpMargin = std::min(400, std::max(0, std::stoi(value)));
            else if (name == "FP") opts.fpMargin = std::min(600, std::max(0, std::stoi(value)));
            else if (name == "LMP") opts.lmp = std::stoi(value) != 0;
            else if (name == "LMR") opts.lmr = std::min(2, std::max(0, std::stoi(value)));
            else if (name == "HistMalus") opts.histMalus = std::stoi(value) != 0;
            else if (name == "HistLMR") opts.histLmr = std::stoi(value) != 0;
            else if (name == "SEE") opts.see = std::min(2, std::max(0, std::stoi(value)));
            else if (name == "NullEval") opts.nullEval = std::stoi(value) != 0;
            else if (name == "NullDepth") opts.nullDepth = std::stoi(value) != 0;
            else if (name == "TimeManage") opts.timeManage = std::stoi(value) != 0;
            else if (name == "TimeMax") opts.timeMax = std::min(1000, std::max(100, std::stoi(value)));
            else if (name == "MoveOverhead") opts.moveOverhead = std::min(5000, std::max(0, std::stoi(value)));
            else if (name == "Aspiration") opts.aspiration = std::min(200, std::max(0, std::stoi(value)));
            else if (name == "Threads") { joinSearch(); opts.threads = std::min(64, std::max(1, std::stoi(value))); }
            else if (name == "EvalParams") {
                if (value.empty() || value == "<empty>") { EP = BUILTIN; rebuildTables(); }
                else if (value == "original") { setDefaultParams(EP); rebuildTables(); }
                else {
                    std::string err;
                    if (!loadParams(value, err)) emit("info string could not load eval params: " + value + " (" + err + ")");
                }
            } else if (name == "EvalFile") {
                if (!loadNet(value)) emit("info string could not load net: " + value);
                else emit("info string loaded NNUE with hidden size " + std::to_string(net.H));
            }
        } else if (cmd == "position") {
            joinSearch();
            setPosition(in);
        } else if (cmd == "go") {
            joinSearch();
            int depth = 64;
            double movetime = 0;
            bool hasTime = false;
            long wtime = -1, btime = -1, winc = 0, binc = 0, movestogo = 0;
            double optimum = 0;
            std::string tok;
            while (in >> tok) {
                if (tok == "depth") in >> depth;
                else if (tok == "movetime") { long ms; in >> ms; movetime = ms / 1000.0; hasTime = true; }
                else if (tok == "wtime") in >> wtime;
                else if (tok == "btime") in >> btime;
                else if (tok == "winc") in >> winc;
                else if (tok == "binc") in >> binc;
                else if (tok == "movestogo") in >> movestogo;
            }
            if (!hasTime && (wtime >= 0 || btime >= 0)) {
                const long mine = gameBoard.sideToMove() == Color::WHITE ? wtime : btime;
                const long inc = gameBoard.sideToMove() == Color::WHITE ? winc : binc;
                const double mtg = movestogo > 0 ? std::min<long>(movestogo, 50) : 30;
                const double avail = std::max(1.0, double(std::max(0L, mine)) - opts.moveOverhead);   // ms
                double opt = std::min(avail / mtg + inc * 0.8, avail * 0.4);                          // planned ms
                opt = std::max(opt, 5.0);
                if (opts.timeManage) {
                    optimum = opt / 1000.0;
                    movetime = std::max(optimum, std::min(avail * 0.5, opt * opts.timeMax / 100.0) / 1000.0);   // hard limit
                } else {
                    movetime = opt / 1000.0;
                }
                hasTime = true;
            }
            stopFlag = false;
            searchThread = std::thread(doGo, gameBoard, depth, movetime, hasTime, opts.noise, optimum);
        } else if (cmd == "stop") {
            stopFlag = true;
            joinSearch();
        } else if (cmd == "evalparams") {
            emit(dumpParams(EP) + "end");
        } else if (cmd == "eval") {
            emit("eval hand " + std::to_string(evalHand(gameBoard)) + " nnue " +
                 std::to_string(net.ok ? evalNet(gameBoard) : 0));
        } else if (cmd == "nnuebench") {
            // Timing breakdown of one NNUE evaluation (needs a net loaded with EvalFile)
            joinSearch();
            if (!net.ok) { emit("info string nnuebench: no net loaded (setoption name EvalFile value <nnue.bin>)"); }
            else {
                const int N = 20000;
                volatile int sink = 0;
                auto ns = [&](auto&& f) {
                    const auto t = Clock::now();
                    for (int i = 0; i < N; ++i) f(i);
                    return std::chrono::duration<double, std::nano>(Clock::now() - t).count() / N;
                };
                static int16_t q0[1024], q1[1024];
                netFullAccQ(gameBoard, 0, kingBucket(gameBoard, 0), q0);
                netFullAccQ(gameBoard, 1, kingBucket(gameBoard, 1), q1);
                const double fullF = ns([&](int) { sink = sink + evalNet(gameBoard); });
                const double fullQ = ns([&](int) { sink = sink + evalNetQ(gameBoard); });
                const double rebuild = ns([&](int i) { netFullAccQ(gameBoard, i & 1, i & 7, q0); sink = sink + q0[0]; });
                const double dense = ns([&](int i) { q0[0] = int16_t(i & 7); sink = sink + netDenseQ(q0, q1); });
                std::ostringstream o;
                o << "info string nnuebench hidden " << net.H << " | float full eval " << int(fullF) << " ns | int16 full eval "
                  << int(fullQ) << " ns | one accumulator rebuild " << int(rebuild) << " ns | dense layers " << int(dense) << " ns";
                emit(o.str());
            }
        } else if (cmd == "bench") {
            joinSearch();
            bench();
        } else if (cmd == "quit") {
            stopFlag = true;
            joinSearch();
            break;
        }
    }
    stopFlag = true;
    joinSearch();
    return 0;
}