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
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace chess;
using Clock = std::chrono::steady_clock;

namespace {

// All output goes through here: the search thread and the command loop both print, and an unsynchronised std::cout can duplicate or garble lines.
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

void initEval() {
    for (int f = 0; f < 8; ++f) FILE_BB[f] = 0x0101010101010101ULL << f;
    for (int f = 0; f < 8; ++f)
        ADJ_FILES[f] = (f > 0 ? FILE_BB[f - 1] : 0) | (f < 7 ? FILE_BB[f + 1] : 0);
    for (int pt = 1; pt <= 6; ++pt)
        for (int sq = 0; sq < 64; ++sq) {
            MGT[0][pt][sq] = MG_VALUE[pt] + MG_PST[pt][sq ^ 56];
            EGT[0][pt][sq] = EG_VALUE[pt] + EG_PST[pt][sq ^ 56];
            MGT[1][pt][sq] = MG_VALUE[pt] + MG_PST[pt][sq];
            EGT[1][pt][sq] = EG_VALUE[pt] + EG_PST[pt][sq];
        }
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
                    if (!(oppp & PASSED_MASK[ci][sq])) { m += PASSED_MG[r]; e += PASSED_EG[r]; }
                    if (!(ownp & ADJ_FILES[f])) { m -= 10; e -= 15; }
                    if (ownp & FRONT_SAME[ci][sq]) { m -= 12; e -= 20; }
                } else if (pt == 4) {
                    uint64_t fm = FILE_BB[sq & 7];
                    if (!(ownp & fm)) {
                        if (!(oppp & fm)) { m += 20; e += 10; }
                        else              { m += 10; e += 5; }
                    }
                }
            }
            if (pt == 3 && popcnt(bb) >= 2) { m += 30; e += 50; }
            else if (pt == 6) m += 9 * popcnt(ownp & SHIELD[ci][ctz(bb)]);
        }
        mg[ci] = m;
        eg[ci] = e;
    }
    if (phase > 24) phase = 24;
    int mgs = mg[0] - mg[1], egs = eg[0] - eg[1];
    int score = (mgs * phase + egs * (24 - phase)) / 24;   // truncates toward 0, like the Python int()
    return bd.sideToMove() == Color::WHITE ? score + 10 : -score + 10;
}

//  NNUE (v2: king-bucketed, both perspectives) - weights from nnue.bin written by Python
//  File: "NNU2" | int32 H | W1[6144*H] b1[H] W2[2H*32] b2[32] W3[32*32] b3[32] W4[32] b4  (float32)
struct Net {
    int H = 0;
    std::vector<float> W1, b1, W2, b2, W3, b3, W4;
    float b4 = 0;
    bool ok = false;
} net;

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
    return net.ok;
}

int evalNet(const Board& bd) {
    const int H = net.H;
    float acc[2][1024];
    const Color stm = bd.sideToMove();
    for (int side = 0; side < 2; ++side) {
        const Color pc = side == 0 ? stm : ~stm;
        const int flip = pc == Color::WHITE ? 0 : 56;
        const int ks = bd.kingSq(pc).index() ^ flip;
        const int bucket = ((ks & 7) >> 1) + ((ks >> 3) >= 2 ? 4 : 0);
        float* a = acc[side];
        std::memcpy(a, net.b1.data(), H * sizeof(float));
        for (int c = 0; c < 2; ++c) {
            const Color col = c == 0 ? Color::WHITE : Color::BLACK;
            for (int t = 0; t < 6; ++t) {
                uint64_t bits = bd.pieces(PTS[t], col).getBits();
                const int base = bucket * 768 + (t + (col == pc ? 0 : 6)) * 64;
                while (bits) {
                    const int sq = ctz(bits);
                    bits &= bits - 1;
                    const float* row = &net.W1[size_t(base + (sq ^ flip)) * H];
                    for (int i = 0; i < H; ++i) a[i] += row[i];
                }
            }
        }
        for (int i = 0; i < H; ++i) a[i] = std::min(1.0f, std::max(0.0f, a[i]));
    }
    float h1[32];
    std::memcpy(h1, net.b2.data(), sizeof(h1));
    for (int side = 0; side < 2; ++side)
        for (int i = 0; i < H; ++i) {
            const float v = acc[side][i];
            if (v == 0.0f) continue;
            const float* w = &net.W2[size_t(side * H + i) * 32];
            for (int j = 0; j < 32; ++j) h1[j] += v * w[j];
        }
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

// Options
struct Options {
    int hashMB = 64;
    bool useNNUE = true;
    int noise = 0;
} opts;

inline int evaluate(const Board& bd) {
    return (opts.useNNUE && net.ok) ? evalNet(bd) : evalHand(bd);
}

// Transposition table
struct TTEntry {
    uint64_t key = 0;
    int16_t score = 0;
    int16_t depth = 0;
    uint8_t flag = 0;      // 0 exact, 1 lower bound, 2 upper bound
    uint16_t move = 0;
};
std::vector<TTEntry> TT;
size_t TT_MASK = 0;

void resizeTT(int mb) {
    size_t n = 1;
    while (n * 2 * sizeof(TTEntry) <= size_t(mb) * 1024 * 1024) n *= 2;
    TT.assign(n, TTEntry());
    TT_MASK = n - 1;
}
inline int toTT(int s, int ply) { return s > MATE - 1000 ? s + ply : (s < -MATE + 1000 ? s - ply : s); }
inline int fromTT(int s, int ply) { return s > MATE - 1000 ? s - ply : (s < -MATE + 1000 ? s + ply : s); }

// Search
struct Scored { Move m; int s; };

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
    std::mt19937_64 rng{std::random_device{}()};

    void reset(bool hasTime, double seconds) {
        nodes = 0;
        stopped = false;
        timed = hasTime;
        if (hasTime) deadline = Clock::now() + std::chrono::microseconds(int64_t(seconds * 1e6));
        std::memset(history, 0, sizeof(history));
        for (auto& k : killers) k[0] = k[1] = Move();
    }

    bool timeUp() {
        if (stopFlag && stopFlag->load(std::memory_order_relaxed)) return true;
        return timed && Clock::now() >= deadline;
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
            } else if (m.typeOf() == Move::PROMOTION) s = 900000 + int(m.promotionType());
            else if (m == killers[ply][0]) s = 800000;
            else if (m == killers[ply][1]) s = 790000;
            else s = history[stm][m.from().index()][m.to().index()];
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
        if (ply >= MAX_PLY) return evaluate(bd);

        const bool inCheck = bd.inCheck();
        Movelist ml;
        int stand;
        if (inCheck && qd < 3) {
            movegen::legalmoves(ml, bd);
            if (ml.empty()) return -MATE + ply;
            stand = -INF;
        } else {
            stand = evaluate(bd);
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
            }
            bd.makeMove(m);
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
        if (ply >= MAX_PLY) return evaluate(bd);

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
        if (e.key == key) {
            ttMove = Move(e.move);
            if (e.depth >= depth) {
                const int s = fromTT(e.score, ply);
                if (e.flag == 0) return s;
                if (e.flag == 1 && s > alpha) alpha = s;
                else if (e.flag == 2 && s < beta) beta = s;
                if (alpha >= beta) return s;
            }
        }

        if (allowNull && !inCheck && depth >= 3 && beta < MATE - 1000 && bd.hasNonPawnMaterial(bd.sideToMove())) {
            bd.makeNullMove();
            const int s = -negamax(depth - 3 - (depth > 6), -beta, -beta + 1, ply + 1, false);
            bd.unmakeNullMove();
            if (stopped) return 0;
            if (s >= beta) return beta;
        }

        Movelist ml;
        movegen::legalmoves(ml, bd);
        if (ml.empty()) return inCheck ? -MATE + ply : 0;
        Scored mv[256];
        const int n = orderMoves(ml, mv, ttMove, ply);
        const int stm = bd.sideToMove() == Color::WHITE ? 0 : 1;

        int best = -INF;
        Move bestMove{Move::NO_MOVE};
        for (int i = 0; i < n; ++i) {
            pickBest(mv, n, i);
            const Move m = mv[i].m;
            const bool quiet = !bd.isCapture(m) && m.typeOf() != Move::PROMOTION;
            bd.makeMove(m);
            int score;
            if (i == 0) {
                score = -negamax(depth - 1, -beta, -alpha, ply + 1, true);
            } else {
                int red = 0;
                if (depth >= 3 && i >= 4 && quiet && !inCheck && !bd.inCheck())
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
                            h = std::min(h + depth * depth, 1000000);
                        }
                        break;
                    }
                }
            }
        }

        const uint8_t flag = best <= origAlpha ? 2 : (best >= beta ? 1 : 0);
        if (e.key != key || depth >= e.depth) {
            e.key = key;
            e.score = int16_t(toTT(best, ply));
            e.depth = int16_t(depth);
            e.flag = flag;
            e.move = bestMove.move();
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

    // Iterative deepening. softFrac: don't start a new depth after this fraction of the time.
    Move iterative(const Movelist& root, int maxDepth, double seconds, bool hasTime, int& scoreOut) {
        reset(hasTime, seconds);
        const auto t0 = Clock::now();
        Move best = root[0];
        int bestScore = 0;
        for (int depth = 1; depth <= maxDepth; ++depth) {
            Scored mv[256];
            const int n = orderMoves(root, mv, best, 0);
            int alpha = -INF;
            Move itBest{Move::NO_MOVE};
            int itScore = -INF;
            for (int i = 0; i < n; ++i) {
                pickBest(mv, n, i);
                const Move m = mv[i].m;
                bd.makeMove(m);
                int s;
                if (i == 0) s = -negamax(depth - 1, -INF, -alpha, 1, true);
                else {
                    s = -negamax(depth - 1, -alpha - 1, -alpha, 1, true);
                    if (s > alpha) s = -negamax(depth - 1, -INF, -alpha, 1, true);
                }
                bd.unmakeMove(m);
                if (stopped) break;
                if (s > itScore) { itScore = s; itBest = m; }
                if (s > alpha) alpha = s;
            }
            if (itBest.move() != 0) { best = itBest; bestScore = itScore; }
            const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
            if (!stopped || itBest.move() != 0) printInfo(depth, bestScore, nodes, secs, best);
            if (stopped || std::abs(bestScore) > MATE - 100) break;
            if (hasTime && secs > seconds * 0.5) break;
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
            bd.makeMove(mv[i].m);
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

void joinSearch() {
    if (searchThread.joinable()) searchThread.join();
}

void doGo(Board board, int depth, double movetime, bool hasTime, int noise) {
    searcher.bd = board;
    searcher.stopFlag = &stopFlag;
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
        best = searcher.iterative(root, depth, movetime, hasTime, score);
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
        std::fill(TT.begin(), TT.end(), TTEntry());
        searcher.iterative(root, 9, 0, false, sc);
        total += searcher.nodes;
    }
    const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
    emit("bench nodes " + std::to_string(total) + " time " + std::to_string(int(secs * 1000)) +
         " nps " + std::to_string(uint64_t(total / secs)));
}

}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    initEval();
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
                 "option name EvalFile type string default <empty>\n"
                 "option name Noise type spin default 0 min 0 max 1000\nuciok");
        } else if (cmd == "isready") {
            emit("readyok");
        } else if (cmd == "ucinewgame") {
            joinSearch();
            std::fill(TT.begin(), TT.end(), TTEntry());
        } else if (cmd == "setoption") {
            std::string tok, name, value;
            in >> tok;  // "name"
            while (in >> tok && tok != "value") name += (name.empty() ? "" : " ") + tok;
            std::getline(in, value);
            while (!value.empty() && value[0] == ' ') value.erase(0, 1);
            if (name == "Hash") { joinSearch(); opts.hashMB = std::max(1, std::stoi(value)); resizeTT(opts.hashMB); }
            else if (name == "UseNNUE") opts.useNNUE = (value == "true" || value == "1");
            else if (name == "Noise") opts.noise = std::max(0, std::stoi(value));
            else if (name == "EvalFile") {
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
            long wtime = -1, btime = -1, winc = 0, binc = 0;
            std::string tok;
            while (in >> tok) {
                if (tok == "depth") in >> depth;
                else if (tok == "movetime") { long ms; in >> ms; movetime = ms / 1000.0; hasTime = true; }
                else if (tok == "wtime") in >> wtime;
                else if (tok == "btime") in >> btime;
                else if (tok == "winc") in >> winc;
                else if (tok == "binc") in >> binc;
            }
            if (!hasTime && (wtime >= 0 || btime >= 0)) {
                long mine = gameBoard.sideToMove() == Color::WHITE ? wtime : btime;
                long inc = gameBoard.sideToMove() == Color::WHITE ? winc : binc;
                movetime = std::max(0.05, (mine / 30.0 + inc * 0.8) / 1000.0);
                hasTime = true;
            }
            stopFlag = false;
            searchThread = std::thread(doGo, gameBoard, depth, movetime, hasTime, opts.noise);
        } else if (cmd == "stop") {
            stopFlag = true;
            joinSearch();
        } else if (cmd == "eval") {
            emit("eval hand " + std::to_string(evalHand(gameBoard)) + " nnue " +
                 std::to_string(net.ok ? evalNet(gameBoard) : 0));
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