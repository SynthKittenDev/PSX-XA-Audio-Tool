import os, sys, io, math, struct, time, wave
import threading, tempfile, subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    import numpy as np
except ImportError:
    np = None

try:
    import sounddevice as sd
    _BACKEND = 'sounddevice' if np is not None else 'os'
    if np is None:
        sd = None
except ImportError:
    sd = None
    _BACKEND = 'os'

_HAS_NUMBA = False
try:
    import numba as _numba
    if np is not None:
        _HAS_NUMBA = True
except ImportError:
    pass

CD_SYNC     = bytes([0x00,0xFF,0xFF,0xFF,0xFF,0xFF,
                     0xFF,0xFF,0xFF,0xFF,0xFF,0x00])
SECTOR_FULL = 2352
SECTOR_RAW  = 2336
AUDIO_BYTES = 2304
N_GROUPS    = 18
GROUP_SZ    = 128
N_UNITS     = 8
SPU         = 28

SM_AUDIO   = 0x04
COD_STEREO = 0x01
COD_18900  = 0x04
COD_8BIT   = 0x10

K0 = [  0,  60, 115,  98, 122]
K1 = [  0,   0, -52, -55, -60]

_EDC_TABLE = None

def _build_edc_table():
    global _EDC_TABLE
    _EDC_TABLE = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = (crc >> 1) ^ 0xD8018001 if (crc & 1) else (crc >> 1)
        _EDC_TABLE.append(crc)

def _calc_edc(data, start: int, length: int) -> int:
    if _EDC_TABLE is None:
        _build_edc_table()
    crc = 0
    for i in range(start, start + length):
        crc = (crc >> 8) ^ _EDC_TABLE[(crc ^ data[i]) & 0xFF]
    return crc

def _nib(n: int, sh: int, f: int, p1: int, p2: int) -> int:
    if n >= 8:
        n -= 16
    t = n << max(0, 12 - sh)
    s = t + (K0[f] * p1 + K1[f] * p2 + 32) // 64
    return max(-32768, min(32767, s))

def _dec_unit(data: bytes, audio_base: int, u: int,
              sh: int, f: int, p1: int, p2: int):
    out       = []
    a, b      = p1, p2
    byte_col  = u >> 1
    nib_shift = (u & 1) << 2
    for d in range(SPU):
        pos  = audio_base + d * 4 + byte_col
        byte = data[pos] if pos < len(data) else 0
        n    = (byte >> nib_shift) & 0xF
        s    = _nib(n, sh, f, a, b)
        b, a = a, s
        out.append(s)
    return out, a, b

def _decode_group(data: bytes, goff: int, stereo: bool, st: list):
    audio_base        = goff + 16
    p1l, p2l, p1r, p2r = st
    L, R = [], []
    for u in range(N_UNITS):
        pidx = u if u < 4 else u + 4
        pb   = data[goff + pidx] if goff + pidx < len(data) else 0
        sh   = pb & 0x0F
        f    = (pb >> 4) & 0x03
        if stereo:
            if u % 2 == 0:
                s, p1l, p2l = _dec_unit(data, audio_base, u, sh, f, p1l, p2l)
                L.extend(s)
            else:
                s, p1r, p2r = _dec_unit(data, audio_base, u, sh, f, p1r, p2r)
                R.extend(s)
        else:
            s, p1l, p2l = _dec_unit(data, audio_base, u, sh, f, p1l, p2l)
            L.extend(s)
    st[:] = [p1l, p2l, p1r, p2r]
    return L, R

def _decode_block(audio: bytes, stereo: bool, st: list) -> list:
    out = []
    for g in range(N_GROUPS):
        L, R = _decode_group(audio, g * GROUP_SZ, stereo, st)
        if stereo:
            for l, r in zip(L, R):
                out += [l, r]
        else:
            out += L
    return out

_ENC_F_LIST  = [f  for f in range(4) for _  in range(13)]
_ENC_SH_LIST = [sh for _  in range(4) for sh in range(13)]
_ENC_COMBOS  = [(K0[f], K1[f], 1 << (12 - sh), sh, f)
                for f in range(4) for sh in range(13)]

_ENC_K0_NP  = None
_ENC_K1_NP  = None
_ENC_DIV_NP = None
_NP_ENC     = None

def _init_enc_tables():
    global _ENC_K0_NP, _ENC_K1_NP, _ENC_DIV_NP, _NP_ENC
    if np is None:
        return
    _ENC_K0_NP  = np.array([K0[f] for f in _ENC_F_LIST],  dtype=np.float64)
    _ENC_K1_NP  = np.array([K1[f] for f in _ENC_F_LIST],  dtype=np.float64)
    _ENC_DIV_NP = np.array([1 << (12 - sh) for sh in _ENC_SH_LIST], dtype=np.float64)
    if not _HAS_NUMBA:
        N = 52
        _NP_ENC = {
            'a'   : np.zeros(N, dtype=np.float64),
            'b'   : np.zeros(N, dtype=np.float64),
            'mse' : np.zeros(N, dtype=np.float64),
            'pred': np.zeros(N, dtype=np.float64),
            'tmp' : np.zeros(N, dtype=np.float64),
            'nib' : np.zeros(N, dtype=np.float64),
            'dec' : np.zeros(N, dtype=np.float64),
            'inim': np.zeros(N, dtype=np.int32),
            'nibs': np.zeros((N, SPU), dtype=np.int32),
        }

if _HAS_NUMBA:
    @_numba.njit(cache=True)
    def _enc_unit_nb_impl(smp, p1d, p2d, K0a, K1a, DIVa):
        N = 52
        a    = np.empty(N, np.float64)
        b    = np.empty(N, np.float64)
        mse  = np.zeros(N,  np.float64)
        nibs = np.empty((N, 28), np.int32)
        for ci in range(N):
            a[ci] = p1d
            b[ci] = p2d
        for si in range(28):
            s = smp[si]
            for ci in range(N):
                pred = (K0a[ci] * a[ci] + K1a[ci] * b[ci] + 32.0) // 64.0
                raw  = (s - pred) / DIVa[ci]
                nb_f = round(raw)
                if nb_f < -8.0:
                    nb_f = -8.0
                elif nb_f > 7.0:
                    nb_f = 7.0
                dec = nb_f * DIVa[ci] + pred
                if dec < -32768.0:
                    dec = -32768.0
                elif dec > 32767.0:
                    dec = 32767.0
                mse[ci] += (s - dec) * (s - dec)
                nibs[ci, si] = int(nb_f) & 0xF
                b[ci] = a[ci]
                a[ci] = dec
        best = 0
        for ci in range(1, N):
            if mse[ci] < mse[best]:
                best = ci
        res = np.empty(28, np.int32)
        for i in range(28):
            res[i] = nibs[best, i]
        return res, best, a[best], b[best]

def _enc_unit(smp, p1: int, p2: int):
    if np is not None:
        if _ENC_K0_NP is None:
            _init_enc_tables()
        if _HAS_NUMBA:
            smp_a = np.array(smp, dtype=np.float64)
            res, best, fa, fb = _enc_unit_nb_impl(
                smp_a, float(p1), float(p2),
                _ENC_K0_NP, _ENC_K1_NP, _ENC_DIV_NP)
            best = int(best)
            return (res.tolist(),
                    int(_ENC_SH_LIST[best]),
                    int(_ENC_F_LIST[best]),
                    int(round(fa)),
                    int(round(fb)))
        E   = _NP_ENC
        a   = E['a'];    b   = E['b'];    mse = E['mse']
        prd = E['pred']; tmp = E['tmp'];  nib = E['nib']
        dec = E['dec'];  nim = E['inim']; nb  = E['nibs']
        K0a = _ENC_K0_NP
        K1a = _ENC_K1_NP
        DIV = _ENC_DIV_NP
        a[:] = p1
        b[:] = p2
        mse[:] = 0.0
        for i, s in enumerate(smp):
            np.multiply(K0a, a,             out=prd)
            np.multiply(K1a, b,             out=tmp)
            np.add(prd, tmp,                out=prd)
            prd += 32.0
            prd //= 64.0
            np.subtract(s, prd,             out=tmp)
            np.divide(tmp, DIV,             out=nib)
            np.round(nib,                   out=nib)
            np.clip(nib, -8.0, 7.0,         out=nib)
            np.multiply(nib, DIV,           out=dec)
            np.add(dec, prd,                out=dec)
            np.clip(dec, -32768.0, 32767.0, out=dec)
            np.subtract(s, dec,             out=tmp)
            np.multiply(tmp, tmp,           out=tmp)
            np.add(mse, tmp,                out=mse)
            nim[:] = nib
            nim  &= 0xF
            nb[:, i] = nim
            b[:] = a
            a[:] = dec
        best = int(np.argmin(mse))
        return (nb[best].tolist(),
                int(_ENC_SH_LIST[best]),
                int(_ENC_F_LIST[best]),
                int(round(a[best])),
                int(round(b[best])))
    _mx = max
    _mn = min
    _rd = round
    best_mse = float('inf')
    best = None
    _nb = [0] * SPU
    for k0, k1, div, sh, f in _ENC_COMBOS:
        a, b = p1, p2
        mse = 0
        for i, s in enumerate(smp):
            pred = (k0 * a + k1 * b + 32) >> 6
            nib  = _mx(-8, _mn(7, _rd((s - pred) / div)))
            dec  = _mx(-32768, _mn(32767, nib * div + pred))
            d    = s - dec
            mse += d * d
            b, a = a, dec
            _nb[i] = nib & 0xF
        if mse < best_mse:
            best_mse = mse
            best = (list(_nb), sh, f, a, b)
            if mse == 0:
                return best
    return best

def _encode_group(L: list, R, st: list) -> bytes:
    stereo = bool(R)
    p1l, p2l, p1r, p2r = st
    grp      = bytearray(GROUP_SZ)
    all_nibs = [None] * N_UNITS

    for u in range(N_UNITS):
        ci = u >> 1
        if stereo:
            if u & 1 == 0:
                chunk = L[ci * SPU:(ci + 1) * SPU]
                if len(chunk) < SPU:
                    chunk += [0] * (SPU - len(chunk))
                nibs, sh, fl, p1l, p2l = _enc_unit(chunk, p1l, p2l)
            else:
                chunk = R[ci * SPU:(ci + 1) * SPU]
                if len(chunk) < SPU:
                    chunk += [0] * (SPU - len(chunk))
                nibs, sh, fl, p1r, p2r = _enc_unit(chunk, p1r, p2r)
        else:
            chunk = L[u * SPU:(u + 1) * SPU]
            if len(chunk) < SPU:
                chunk += [0] * (SPU - len(chunk))
            nibs, sh, fl, p1l, p2l = _enc_unit(chunk, p1l, p2l)

        all_nibs[u] = nibs
        pb = (fl << 4) | (sh & 0xF)

        if u < 4:
            grp[u]     = pb
            grp[u + 4] = pb
        else:
            grp[u + 4] = pb
            grp[u + 8] = pb

    for u in range(N_UNITS):
        nibs      = all_nibs[u]
        byte_col  = u >> 1
        nib_shift = (u & 1) << 2
        for d in range(SPU):
            pos = 16 + d * 4 + byte_col
            grp[pos] |= (nibs[d] & 0xF) << nib_shift

    st[:] = [p1l, p2l, p1r, p2r]
    return bytes(grp)

def encode_xa(pcm: list, stereo: bool) -> list:
    spg = 4 * SPU if stereo else 8 * SPU
    spb = N_GROUPS * spg
    L = list(pcm[0::2]) if stereo else list(pcm)
    R = list(pcm[1::2]) if stereo else None
    pad = (-len(L)) % spb
    if pad:
        L += [0] * pad
    if stereo:
        pad = (-len(R)) % spb
        if pad:
            R += [0] * pad
    n_blk  = len(L) // spb
    blocks = [None] * n_blk
    st     = [0, 0, 0, 0]
    for bi in range(n_blk):
        buf  = bytearray(AUDIO_BYTES)
        base = bi * N_GROUPS
        for g in range(N_GROUPS):
            gi  = base + g
            Lc  = L[gi * spg:(gi + 1) * spg]
            Rc  = R[gi * spg:(gi + 1) * spg] if stereo else None
            buf[g * GROUP_SZ:(g + 1) * GROUP_SZ] = _encode_group(Lc, Rc, st)
        blocks[bi] = bytes(buf)
    return blocks

def read_wav(path: str):
    with wave.open(path, 'rb') as w:
        ch, sw, rate, nf = (w.getnchannels(), w.getsampwidth(),
                            w.getframerate(), w.getnframes())
        raw = w.readframes(nf)
    bits, n = sw * 8, nf * ch
    if   bits == 16: pcm = list(struct.unpack(f'<{n}h', raw))
    elif bits ==  8: pcm = [(b - 128) << 8 for b in raw]
    elif bits == 24:
        pcm = []
        for i in range(0, len(raw), 3):
            v = raw[i] | (raw[i+1] << 8) | (raw[i+2] << 16)
            if v >= 0x800000: v -= 0x1000000
            pcm.append(v >> 8)
    elif bits == 32:
        pcm = [v >> 16 for v in struct.unpack(f'<{n}i', raw)]
    else:
        raise ValueError(f"Unsupported WAV bit depth: {bits}")
    return pcm, ch, rate, bits

def pcm_to_wav_bytes(pcm: list, ch: int, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(ch); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(struct.pack(f'<{len(pcm)}h', *pcm))
    return buf.getvalue()

def write_wav(path: str, pcm: list, ch: int, rate: int):
    with open(path, 'wb') as f:
        f.write(pcm_to_wav_bytes(pcm, ch, rate))

def _sinc_lowpass(cutoff: float, n_taps: int = 127) -> list:
    half = n_taps // 2
    h = []
    for i in range(n_taps):
        t    = i - half
        sinc = (2.0 * cutoff
                if t == 0
                else math.sin(2.0 * math.pi * cutoff * t) / (math.pi * t))
        win  = 0.54 - 0.46 * math.cos(2.0 * math.pi * i / (n_taps - 1))
        h.append(sinc * win)
    total = sum(h)
    return [v / total for v in h]

def _fir_filter(s: list, h: list) -> list:
    if np is not None:
        sig   = np.array(s, dtype=np.float64)
        fil   = np.array(h, dtype=np.float64)
        n_sig = len(sig)
        n_fil = len(fil)
        blk   = 1 << max(10, (n_fil * 8 - 1).bit_length())
        nfft  = 1 << (blk + n_fil - 2).bit_length()
        H     = np.fft.rfft(fil, nfft)
        out   = np.zeros(n_sig + n_fil - 1, dtype=np.float64)
        for pos in range(0, n_sig, blk):
            block = sig[pos:pos + blk]
            y     = np.fft.irfft(np.fft.rfft(block, nfft) * H, nfft)
            end   = min(pos + nfft, len(out))
            out[pos:end] += y[:end - pos]
        half = n_fil // 2
        out  = out[half:half + n_sig]
        return np.clip(np.round(out), -32768, 32767).astype(np.int32).tolist()
    n_h    = len(h)
    half   = n_h // 2
    pad    = [0] * half
    padded = pad + list(s) + pad
    out    = []
    for i in range(len(s)):
        v = sum(padded[i + j] * h[j] for j in range(n_h))
        out.append(max(-32768, min(32767, int(round(v)))))
    return out

def _mono_resample(s: list, f_in: int, f_out: int) -> list:
    if f_in == f_out:
        return list(s)
    if f_out < f_in:
        cutoff = (f_out / 2.0) / f_in
        s = _fir_filter(s, _sinc_lowpass(cutoff))
    if np is not None:
        sig = np.asarray(s, dtype=np.float64)
        n   = int(round(len(sig) * f_out / f_in))
        pos = np.arange(n, dtype=np.float64) * (f_in / f_out)
        j   = pos.astype(np.intp)
        fr  = pos - j
        j0  = np.clip(j,     0, len(sig) - 1)
        j1  = np.clip(j + 1, 0, len(sig) - 1)
        out = sig[j0] + fr * (sig[j1] - sig[j0])
        return np.clip(np.round(out), -32768, 32767).astype(np.int32).tolist()
    n     = int(round(len(s) * f_out / f_in))
    ratio = f_in / f_out
    out   = []
    for i in range(n):
        pos = i * ratio
        j   = int(pos)
        fr  = pos - j
        a   = s[j]     if j     < len(s) else 0
        b   = s[j + 1] if j + 1 < len(s) else 0
        out.append(max(-32768, min(32767, int(round(a + fr * (b - a))))))
    return out

def resample(pcm: list, f_in: int, f_out: int, ch: int) -> list:
    if f_in == f_out:
        return pcm
    if ch == 1:
        return _mono_resample(pcm, f_in, f_out)
    L = _mono_resample(pcm[0::2], f_in, f_out)
    R = _mono_resample(pcm[1::2], f_in, f_out)
    if np is not None:
        La  = np.array(L, dtype=np.int32)
        Ra  = np.array(R, dtype=np.int32)
        out = np.empty(len(La) + len(Ra), dtype=np.int32)
        out[0::2] = La
        out[1::2] = Ra
        return out.tolist()
    out = []
    for l, r in zip(L, R):
        out += [l, r]
    return out

def to_channels(pcm: list, fc: int, tc: int) -> list:
    if fc == tc:
        return pcm
    if fc == 1:
        return [v for s in pcm for v in (s, s)]
    return [(pcm[i] + pcm[i + 1]) // 2 for i in range(0, len(pcm), 2)]

def _detect_fmt(data: bytes):
    if len(data) >= 12 and data[:12] == CD_SYNC:
        return 'full', SECTOR_FULL
    if len(data) > 0:
        if len(data) % SECTOR_FULL == 0: return 'full', SECTOR_FULL
        if len(data) % SECTOR_RAW  == 0: return 'raw',  SECTOR_RAW
        if len(data) % AUDIO_BYTES == 0: return 'bare', AUDIO_BYTES
    return 'raw', SECTOR_RAW

def scan_file(path: str):
    with open(path, 'rb') as f:
        raw = f.read()
    fmt, ss = _detect_fmt(raw)
    if fmt == 'bare':
        n    = len(raw) // AUDIO_BYTES
        secs = [{'offset': i * AUDIO_BYTES, 'coding': COD_STEREO,
                 'file_num': 0, 'channel_num': 0} for i in range(n)]
        return raw, [{
            'index': 0, 'key': (0, 0), 'file_num': 0, 'channel_num': 0,
            'coding': COD_STEREO, 'stereo': True, 'rate': 37800,
            'has_8bit_flag': False, 'sectors': secs, 'n_sectors': n,
        }], fmt, ss
    sub_off     = 16 if fmt == 'full' else 4
    track_map   = {}
    track_order = []
    off = 0
    while off + ss <= len(raw):
        if off + sub_off + 4 <= len(raw):
            sm = raw[off + sub_off + 2]
            if sm & SM_AUDIO:
                fn   = raw[off + sub_off + 0]
                ch_n = raw[off + sub_off + 1]
                cod  = raw[off + sub_off + 3]
                key  = (fn, ch_n)
                if key not in track_map:
                    track_map[key] = []
                    track_order.append(key)
                track_map[key].append({
                    'offset': off, 'coding': cod,
                    'file_num': fn, 'channel_num': ch_n,
                })
        off += ss
    tracks = []
    for i, key in enumerate(track_order):
        fn, ch_n = key
        secs = track_map[key]
        cod  = secs[0]['coding']
        tracks.append({
            'index': i, 'key': key,
            'file_num': fn, 'channel_num': ch_n,
            'coding': cod,
            'stereo':        bool(cod & COD_STEREO),
            'rate':          18900 if (cod & COD_18900) else 37800,
            'has_8bit_flag': bool(cod & COD_8BIT),
            'sectors': secs, 'n_sectors': len(secs),
        })
    if not tracks:
        secs, off = [], 0
        while off + ss <= len(raw):
            cod = (raw[off + sub_off + 3]
                   if off + sub_off + 4 <= len(raw) else COD_STEREO)
            secs.append({'offset': off, 'coding': cod,
                         'file_num': 0, 'channel_num': 0})
            off += ss
        if secs:
            cod = secs[0]['coding']
            tracks = [{
                'index': 0, 'key': (0, 0), 'file_num': 0, 'channel_num': 0,
                'coding': cod, 'stereo': bool(cod & COD_STEREO),
                'rate': 18900 if (cod & COD_18900) else 37800,
                'has_8bit_flag': bool(cod & COD_8BIT),
                'sectors': secs, 'n_sectors': len(secs),
            }]
    return raw, tracks, fmt, ss

def _aud_rel(fmt: str) -> int:
    return 24 if fmt == 'full' else (0 if fmt == 'bare' else 12)

def decode_track(raw: bytes, track: dict, stereo: bool, fmt: str) -> list:
    ar  = _aud_rel(fmt)
    st  = [0, 0, 0, 0]
    pcm = []
    for sec in track['sectors']:
        ao    = sec['offset'] + ar
        audio = raw[ao:ao + AUDIO_BYTES]
        if len(audio) < AUDIO_BYTES:
            audio += bytes(AUDIO_BYTES - len(audio))
        pcm += _decode_block(audio, stereo, st)
    return pcm

def rebuild_bin(raw: bytes, track: dict, blks: list, fmt: str) -> bytes:
    data   = bytearray(raw)
    ar     = _aud_rel(fmt)
    coding = track['coding']
    for sec, blk in zip(track['sectors'], blks):
        off = sec['offset']
        data[off + ar : off + ar + AUDIO_BYTES] = blk[:AUDIO_BYTES]
        if fmt == 'full':
            data[off + 19] = coding
            data[off + 23] = coding
            edc = _calc_edc(data, off + 0x010, 0x91C)
            struct.pack_into('<I', data, off + 0x92C, edc)
        elif fmt == 'raw':
            data[off + 7]  = coding
            data[off + 11] = coding
    return bytes(data)

def overwrite_track(path: str, track: dict, blks: list, fmt: str):
    ar     = _aud_rel(fmt)
    coding = track['coding']
    with open(path, 'r+b') as f:
        for sec, blk in zip(track['sectors'], blks):
            off = sec['offset']
            if fmt == 'full':
                f.seek(off)
                sector = bytearray(f.read(SECTOR_FULL))
                sector[ar : ar + AUDIO_BYTES] = blk[:AUDIO_BYTES]
                sector[19] = coding
                sector[23] = coding
                edc = _calc_edc(sector, 0x010, 0x91C)
                struct.pack_into('<I', sector, 0x92C, edc)
                f.seek(off)
                f.write(sector)
            else:
                f.seek(off + ar)
                f.write(blk[:AUDIO_BYTES])
                if fmt == 'raw':
                    f.seek(off + 7);  f.write(bytes([coding]))
                    f.seek(off + 11); f.write(bytes([coding]))

_stop_evt = threading.Event()
_play_thr = None

def stop_play():
    global _play_thr
    _stop_evt.set()
    if _BACKEND == 'sounddevice' and sd:
        try: sd.stop()
        except: pass
    if _play_thr: _play_thr.join(timeout=3)
    _stop_evt.clear()

def play_pcm(pcm: list, ch: int, rate: int, done_cb=None, get_volume=None):
    global _play_thr
    stop_play()
    def work():
        try:
            if _BACKEND == 'sounddevice' and np and sd:
                arr = np.array(pcm, dtype=np.float32) / 32768.0
                arr = arr.reshape(-1, ch)
                pos = [0]
                def callback(outdata, frames, time_info, status):
                    vol = get_volume() if get_volume else 1.0
                    start = pos[0]
                    if start >= len(arr) or _stop_evt.is_set():
                        outdata[:] = 0
                        raise sd.CallbackStop()
                    end = start + frames
                    chunk = arr[start:end]
                    n = len(chunk)
                    outdata[:n] = chunk * vol
                    if n < frames:
                        outdata[n:] = 0
                        pos[0] = start + n
                        raise sd.CallbackStop()
                    pos[0] = end
                with sd.OutputStream(samplerate=rate, channels=ch,
                                     dtype='float32', callback=callback) as stream:
                    while stream.active:
                        if _stop_evt.is_set(): break
                        time.sleep(0.05)
            else:
                vol = get_volume() if get_volume else 1.0
                if abs(vol - 1.0) > 1e-6:
                    play_data = [max(-32768, min(32767, int(s * vol))) for s in pcm]
                else:
                    play_data = pcm
                tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                tmp.write(pcm_to_wav_bytes(play_data, ch, rate))
                tmp.close()
                if sys.platform == 'win32':
                    import winsound
                    winsound.PlaySound(tmp.name, winsound.SND_FILENAME)
                elif sys.platform == 'darwin':
                    p = subprocess.Popen(['afplay', tmp.name])
                    while p.poll() is None:
                        if _stop_evt.is_set(): p.kill(); break
                        time.sleep(0.05)
                else:
                    for pl in ['aplay', 'paplay', 'ffplay', 'mpv', 'cvlc']:
                        try:
                            p = subprocess.Popen([pl, tmp.name],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
                            while p.poll() is None:
                                if _stop_evt.is_set(): p.kill(); break
                                time.sleep(0.05)
                            break
                        except FileNotFoundError: continue
                try: os.unlink(tmp.name)
                except: pass
        finally:
            if done_cb: done_cb()
    _play_thr = threading.Thread(target=work, daemon=True)
    _play_thr.start()

class XATool(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PSX XA Audio Tool  v1.0")
        self.minsize(940, 700)
        try: ttk.Style(self).theme_use('clam')
        except: pass
        self.source_path  = None
        self.source_raw   = None
        self.tracks       = []
        self.fmt          = 'raw'
        self.ss           = SECTOR_RAW
        self.sel_track    = None
        self.decoded_pcm  = []
        self.decoded_ch   = 2
        self.decoded_rate = 37800
        self.wav_pcm  = []
        self.wav_ch   = 1
        self.wav_rate = 44100
        self.v_ch   = tk.IntVar(value=2)
        self.v_rate = tk.IntVar(value=37800)
        self._vol   = [1.0]
        self.bulk_queue        = []
        self._bulk_iid_counter = 0
        self._play_gen         = 0
        self._tree_sort        = {'col': None, 'reverse': False}
        self._bulk_sort        = {'col': None, 'reverse': False}
        self._build_ui()

    def _build_ui(self):
        P = dict(padx=8, pady=4)
        fg = ttk.LabelFrame(self, text=" BIN / XA Source File ")
        fg.pack(fill='x', **P)
        rg = ttk.Frame(fg); rg.pack(fill='x', padx=6, pady=4)
        ttk.Button(rg, text="Open BIN / XA…",
                   command=self.open_file).pack(side='left')
        self.l_src = ttk.Label(rg, text="(no file loaded)",
                               relief='sunken', anchor='w')
        self.l_src.pack(side='left', fill='x', expand=True, padx=6)
        self.l_src_info = ttk.Label(fg, text="", foreground='#555', anchor='w')
        self.l_src_info.pack(fill='x', padx=8, pady=(0, 4))

        ft = ttk.LabelFrame(self,
                            text=" XA Tracks  (each file+channel pair is a separate stream) ")
        ft.pack(fill='x', **P)
        cols   = ('#', 'File', 'Channel', 'Channels', 'Rate', 'Sectors', 'Duration', 'Notes')
        widths = [ 28,   52,      64,        74,         80,     64,        80,          150]
        self.tree = ttk.Treeview(ft, columns=cols, show='headings',
                                 height=5, selectmode='browse')
        for c, w in zip(cols, widths):
            self.tree.heading(c, text=c,
                              command=lambda _c=c: self._sort_tree(
                                  self.tree, _c, self._tree_sort))
            self.tree.column(c, width=w, anchor='center', stretch=(c == 'Notes'))
        vsb = ttk.Scrollbar(ft, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side='right', fill='y')
        self.tree.pack(fill='x', padx=4, pady=(4, 2))
        self.tree.bind('<<TreeviewSelect>>', self._on_track_select)

        ro = ttk.Frame(ft); ro.pack(fill='x', padx=6, pady=(2, 6))
        ttk.Label(ro, text="Decode override:").pack(side='left')
        ttk.Label(ro, text="Channels:").pack(side='left', padx=(10, 2))
        ttk.Radiobutton(ro, text="Stereo", variable=self.v_ch, value=2,
                        command=self._redecode).pack(side='left', padx=2)
        ttk.Radiobutton(ro, text="Mono", variable=self.v_ch, value=1,
                        command=self._redecode).pack(side='left', padx=2)
        ttk.Label(ro, text="Rate:").pack(side='left', padx=(10, 2))
        ttk.Radiobutton(ro, text="37800 Hz", variable=self.v_rate, value=37800,
                        command=self._redecode).pack(side='left', padx=2)
        ttk.Radiobutton(ro, text="18900 Hz", variable=self.v_rate, value=18900,
                        command=self._redecode).pack(side='left', padx=2)
        self.l_dec = ttk.Label(ro, text="← click a track row to decode it",
                               foreground='#888')
        self.l_dec.pack(side='left', padx=14)

        fp = ttk.LabelFrame(self, text=" Playback ")
        fp.pack(fill='x', **P)
        rp = ttk.Frame(fp); rp.pack(fill='x', padx=6, pady=4)
        self.b_play = ttk.Button(rp, text="▶  Play Track",
                                 command=self.play_track, state='disabled')
        self.b_stop = ttk.Button(rp, text="⏹  Stop",
                                 command=self.do_stop, state='disabled')
        self.b_exp  = ttk.Button(rp, text="💾  Export WAV…",
                                 command=self.export_wav, state='disabled')
        for b in (self.b_play, self.b_stop, self.b_exp): b.pack(side='left', padx=4)
        ttk.Label(rp, text="Volume:").pack(side='left', padx=(12, 2))
        self.vol_slider = ttk.Scale(rp, from_=0.0, to=1.0, orient='horizontal',
                                    length=120,
                                    command=lambda v: self._vol.__setitem__(0, float(v)))
        self.vol_slider.set(1.0)
        self.vol_slider.pack(side='left', padx=(0, 4))
        self.pbar = ttk.Progressbar(fp, mode='indeterminate')
        self.pbar.pack(fill='x', padx=8, pady=(2, 6))

        fr = ttk.LabelFrame(self, text=" Replace Selected Track ")
        fr.pack(fill='x', **P)
        rr = ttk.Frame(fr); rr.pack(fill='x', padx=6, pady=4)
        ttk.Button(rr, text="Load WAV…", command=self.load_wav).pack(side='left')
        self.l_wav = ttk.Label(rr, text="(no WAV loaded)",
                               relief='sunken', anchor='w')
        self.l_wav.pack(side='left', fill='x', expand=True, padx=6)
        self.l_wav_info = ttk.Label(
            fr,
            text="Accepts any sample rate — "
                 "auto-resampled to the XA track's native rate on replace.",
            foreground='#555', anchor='w')
        self.l_wav_info.pack(fill='x', padx=8)
        rb = ttk.Frame(fr); rb.pack(fill='x', padx=6, pady=4)
        self.b_prv = ttk.Button(rb, text="▶  Preview WAV",
                                command=self.prev_wav, state='disabled')
        self.b_rep = ttk.Button(rb, text="💾  Replace & Save As…",
                                command=lambda: self.do_replace(overwrite=False),
                                state='disabled')
        self.b_ovr = ttk.Button(rb, text="⚠  Overwrite Original",
                                command=lambda: self.do_replace(overwrite=True),
                                state='disabled')
        for b in (self.b_prv, self.b_rep, self.b_ovr): b.pack(side='left', padx=4)
        ttk.Label(rb,
                  text="← 'Overwrite' writes audio sectors in-place; "
                       "all other data is untouched",
                  foreground='#888').pack(side='left', padx=6)

        fb = ttk.LabelFrame(self, text=" Bulk Replace ")
        fb.pack(fill='x', **P)
        bcols   = ('Track', 'File', 'Ch', 'Channels', 'Rate', 'WAV File', 'Status')
        bwidths = [  48,     52,    52,      70,         80,     270,        70  ]
        self.bulk_tree = ttk.Treeview(fb, columns=bcols, show='headings',
                                      height=4, selectmode='extended')
        for c, w in zip(bcols, bwidths):
            self.bulk_tree.heading(c, text=c,
                                   command=lambda _c=c: self._sort_tree(
                                       self.bulk_tree, _c, self._bulk_sort))
            self.bulk_tree.column(c, width=w, anchor='center',
                                  stretch=(c == 'WAV File'))
        bvsb = ttk.Scrollbar(fb, orient='vertical', command=self.bulk_tree.yview)
        self.bulk_tree.configure(yscrollcommand=bvsb.set)
        bvsb.pack(side='right', fill='y')
        self.bulk_tree.pack(fill='x', padx=4, pady=(4, 2))
        rb2 = ttk.Frame(fb); rb2.pack(fill='x', padx=6, pady=4)
        self.b_badd = ttk.Button(rb2, text="+ Add to Queue",
                                 command=self._add_to_queue, state='disabled')
        self.b_brem = ttk.Button(rb2, text="− Remove",
                                 command=self._remove_from_queue, state='disabled')
        self.b_bclr = ttk.Button(rb2, text="✕ Clear Queue",
                                 command=self._clear_queue, state='disabled')
        self.b_bsav = ttk.Button(rb2, text="💾  Bulk Save As…",
                                 command=lambda: self.bulk_replace(overwrite=False),
                                 state='disabled')
        self.b_bovr = ttk.Button(rb2, text="⚠  Bulk Overwrite",
                                 command=lambda: self.bulk_replace(overwrite=True),
                                 state='disabled')
        for b in (self.b_badd, self.b_brem, self.b_bclr, self.b_bsav, self.b_bovr):
            b.pack(side='left', padx=4)
        ttk.Label(rb2,
                  text="← select a track + WAV above, add to queue, then run all at once",
                  foreground='#888').pack(side='left', padx=6)

        fl = ttk.LabelFrame(self, text=" Log ")
        fl.pack(fill='both', expand=True, **P)
        self.log = tk.Text(fl, state='disabled', height=7,
                           font=('Courier', 9), wrap='word',
                           bg='#1c1c2e', fg='#cdd6f4')
        sb = ttk.Scrollbar(fl, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        self.log.pack(fill='both', expand=True, padx=4, pady=4)

        self.sbar = ttk.Label(self, text="Ready.", anchor='w', relief='sunken')
        self.sbar.pack(fill='x', side='bottom', padx=4, pady=2)

        numpy_status = 'yes' if np is not None else 'NO — install numpy for fast resampling'
        enc_mode = ('numba' if _HAS_NUMBA else 'numpy') if np is not None else 'python'
        self._log(f"PSX XA Audio Tool v1.0  |  backend: {_BACKEND}  |"
                  f"  numpy: {numpy_status}  |  encoder: {enc_mode}")
        self._log("Open a .bin or .xa file.  "
                  "Each (file, channel) XA stream appears as a separate track row.")

    def _sort_tree(self, tree, col, sort_state):
        reverse = not sort_state.get('reverse', False) if sort_state.get('col') == col else False
        sort_state['col'] = col
        sort_state['reverse'] = reverse

        def key(iid):
            v = tree.set(iid, col)
            if col in ('#', 'Track', 'Sectors'):
                try: return int(v)
                except: return 0
            if col in ('File', 'Channel', 'Ch'):
                try: return int(v, 16)
                except: return 0
            if col == 'Rate':
                try: return int(v.replace(' Hz', '').strip())
                except: return 0
            if col == 'Duration':
                try: return float(v.rstrip('s').strip())
                except: return 0.0
            return v.lower()

        data = [(key(iid), iid) for iid in tree.get_children('')]
        data.sort(reverse=reverse)
        for i, (_, iid) in enumerate(data):
            tree.move(iid, '', i)

        for c in tree['columns']:
            txt = tree.heading(c, 'text')
            for sfx in (' ▲', ' ▼'):
                if txt.endswith(sfx):
                    txt = txt[:-len(sfx)]
                    break
            arrow = (' ▲' if not reverse else ' ▼') if c == col else ''
            tree.heading(c, text=txt + arrow)

    def _log(self, msg: str):
        self.log.configure(state='normal')
        self.log.insert('end', msg + '\n')
        self.log.see('end')
        self.log.configure(state='disabled')
        self.sbar.configure(text=msg[:120])
        self.update_idletasks()

    def _busy(self, on: bool):
        if on: self.pbar.start(12)
        else:  self.pbar.stop()

    def _upd_btns(self):
        has_pcm  = bool(self.decoded_pcm)
        has_both = has_pcm and bool(self.wav_pcm)
        self.b_play.configure(state='normal' if has_pcm  else 'disabled')
        self.b_exp.configure( state='normal' if has_pcm  else 'disabled')
        self.b_rep.configure( state='normal' if has_both else 'disabled')
        self.b_ovr.configure( state='normal'
                              if (has_both and self.source_path) else 'disabled')
        self._upd_bulk_btns()

    def _upd_bulk_btns(self):
        has_both  = bool(self.sel_track) and bool(self.wav_pcm)
        has_queue = bool(self.bulk_queue)
        self.b_badd.configure(state='normal' if has_both  else 'disabled')
        self.b_brem.configure(state='normal' if has_queue else 'disabled')
        self.b_bclr.configure(state='normal' if has_queue else 'disabled')
        self.b_bsav.configure(state='normal' if has_queue else 'disabled')
        self.b_bovr.configure(
            state='normal' if (has_queue and self.source_path) else 'disabled')

    def open_file(self):
        p = filedialog.askopenfilename(
            title="Open PSX BIN or XA file",
            filetypes=[("BIN / XA", "*.bin *.BIN *.xa *.XA"),
                       ("All files", "*.*")])
        if p:
            self._busy(True)
            threading.Thread(target=self._do_open, args=(p,), daemon=True).start()

    def _do_open(self, path: str):
        try:
            self.after(0, lambda: self._log(f"Scanning {os.path.basename(path)}…"))
            raw, tracks, fmt, ss = scan_file(path)
            self.source_path = path
            self.source_raw  = raw
            self.tracks      = tracks
            self.fmt         = fmt
            self.ss          = ss
            self.sel_track   = None
            self.decoded_pcm = []
            sz   = len(raw) / (1024 * 1024)
            info = (f"{sz:.1f} MB  |  Format: {fmt.upper()}  "
                    f"|  {ss} B/sector  |  {len(tracks)} XA track(s) found")
            self.after(0, lambda: self.l_src.configure(text=os.path.basename(path)))
            self.after(0, lambda: self.l_src_info.configure(text=info))
            self.after(0, lambda: self._log(
                f"  ✔ {len(tracks)} XA track(s).  Click a row to decode it."))
            self.after(0, self._fill_tree)
            self.after(0, self._upd_btns)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Open Error", str(e)))
            self.after(0, lambda: self._log(f"  ✗ {e}"))
        finally:
            self.after(0, lambda: self._busy(False))

    def _fill_tree(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        self._tree_sort = {'col': None, 'reverse': False}
        for c in self.tree['columns']:
            txt = self.tree.heading(c, 'text')
            for sfx in (' ▲', ' ▼'):
                if txt.endswith(sfx):
                    txt = txt[:-len(sfx)]
                    break
            self.tree.heading(c, text=txt)
        for t in self.tracks:
            spg = 4 * SPU if t['stereo'] else 8 * SPU
            dur = t['n_sectors'] * N_GROUPS * spg / t['rate']
            note = '⚠ 8-bit flag (4-bit decoder used)' if t['has_8bit_flag'] else ''
            self.tree.insert('', 'end', iid=str(t['index']), values=(
                t['index'] + 1,
                f"0x{t['file_num']:02X}",
                f"0x{t['channel_num']:02X}",
                'Stereo' if t['stereo'] else 'Mono',
                f"{t['rate']} Hz",
                t['n_sectors'],
                f"{dur:.2f}s",
                note,
            ))
        self.l_dec.configure(text="← click a track row to decode it")

    def _on_track_select(self, _event=None):
        sel = self.tree.selection()
        if not sel: return
        track = self.tracks[int(sel[0])]
        if track is self.sel_track and self.decoded_pcm:
            return
        self.sel_track   = track
        self.decoded_pcm = []
        self.v_ch.set(2 if track['stereo'] else 1)
        self.v_rate.set(track['rate'])
        self._upd_bulk_btns()
        self._busy(True)
        threading.Thread(
            target=self._do_decode,
            args=(track, self.v_ch.get(), self.v_rate.get()),
            daemon=True).start()

    def _redecode(self):
        if self.sel_track:
            self.decoded_pcm = []
            self._busy(True)
            threading.Thread(
                target=self._do_decode,
                args=(self.sel_track, self.v_ch.get(), self.v_rate.get()),
                daemon=True).start()

    def _do_decode(self, track, ch: int, rate: int):
        try:
            self.after(0, lambda: self._log(
                f"Decoding track {track['index']+1}  "
                f"[File 0x{track['file_num']:02X} / "
                f"Ch 0x{track['channel_num']:02X}]  "
                f"{track['n_sectors']} sector(s)…"))
            if track.get('has_8bit_flag'):
                self.after(0, lambda: self._log(
                    "  ⚠ 8-bit ADPCM flag detected; "
                    "tool uses 4-bit decoder — audio may be incorrect."))
            self.after(0, lambda: self._log(
                f"  ℹ Coding byte: 0x{track['coding']:02X}  "
                f"({'stereo' if track['stereo'] else 'mono'}, "
                f"{track['rate']} Hz)"))
            pcm = decode_track(self.source_raw, track, stereo=(ch == 2), fmt=self.fmt)
            if self.sel_track is not track:
                return
            self.decoded_pcm  = pcm
            self.decoded_ch   = ch
            self.decoded_rate = rate
            dur  = len(pcm) / ch / rate
            info = f"{'Stereo' if ch==2 else 'Mono'}  {rate} Hz  {dur:.2f}s"
            self.after(0, lambda: self.l_dec.configure(text=info))
            self.after(0, lambda: self._log(f"  ✔ {len(pcm):,} samples  ({dur:.2f}s)"))
            if self.wav_pcm and self.wav_rate != rate:
                self.after(0, lambda: self._log(
                    f"  ℹ Loaded WAV ({self.wav_rate} Hz) will be "
                    f"resampled → {rate} Hz when replacing"))
            self.after(0, self._upd_btns)
        except Exception as e:
            self.after(0, lambda: self._log(f"  ✗ Decode error: {e}"))
        finally:
            self.after(0, lambda: self._busy(False))

    def play_track(self):
        if not self.decoded_pcm: return
        t   = self.sel_track
        dur = len(self.decoded_pcm) / self.decoded_ch / self.decoded_rate
        self._log(f"▶ Track {t['index']+1}  "
                  f"[0x{t['file_num']:02X} / 0x{t['channel_num']:02X}]  {dur:.1f}s…")
        self._play_gen += 1
        gen = self._play_gen
        self.b_play.configure(state='disabled')
        self.b_stop.configure(state='normal')
        play_pcm(self.decoded_pcm, self.decoded_ch, self.decoded_rate,
                 done_cb=lambda: self.after(0, lambda: self._play_done(gen)),
                 get_volume=lambda: self._vol[0])

    def prev_wav(self):
        if not self.wav_pcm: return
        self._log("▶ WAV preview…")
        self._play_gen += 1
        gen = self._play_gen
        self.b_prv.configure(state='disabled')
        self.b_stop.configure(state='normal')
        play_pcm(self.wav_pcm, self.wav_ch, self.wav_rate,
                 done_cb=lambda: self.after(0, lambda: self._wav_preview_done(gen)),
                 get_volume=lambda: self._vol[0])

    def do_stop(self):
        self._play_gen += 1
        gen = self._play_gen
        stop_play()
        self.after(150, lambda: self._play_done(gen))

    def _play_done(self, gen=None):
        if gen is not None and gen != self._play_gen:
            return
        self.b_stop.configure(state='disabled')
        self.b_prv.configure(state='normal' if self.wav_pcm else 'disabled')
        self._upd_btns()
        self._log("  Stopped.")

    def _wav_preview_done(self, gen=None):
        if gen is not None and gen != self._play_gen:
            return
        self.b_stop.configure(state='disabled')
        self.b_prv.configure(state='normal' if self.wav_pcm else 'disabled')
        self._upd_btns()
        self._log("  Preview done.")

    def export_wav(self):
        if not self.decoded_pcm: return
        t = self.sel_track
        def_name = (f"track{t['index']+1}"
                    f"_f{t['file_num']:02x}"
                    f"_ch{t['channel_num']:02x}.wav")
        p = filedialog.asksaveasfilename(
            title="Export track as WAV",
            initialfile=def_name,
            defaultextension='.wav',
            filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")])
        if not p: return
        try:
            write_wav(p, self.decoded_pcm, self.decoded_ch, self.decoded_rate)
            self._log(f"  ✔ Exported: {p}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def load_wav(self):
        p = filedialog.askopenfilename(
            title="Load replacement WAV  (any sample rate accepted)",
            filetypes=[("WAV audio", "*.wav *.WAV"), ("All files", "*.*")])
        if not p: return
        try:
            pcm, ch, rate, bits = read_wav(p)
            self.wav_pcm  = pcm
            self.wav_ch   = ch
            self.wav_rate = rate
            self.l_wav.configure(text=os.path.basename(p))
            dur  = len(pcm) / ch / rate
            info = (f"{'Stereo' if ch==2 else 'Mono'}  |  {rate} Hz  |  "
                    f"{bits}-bit  |  {dur:.2f}s")
            if self.decoded_pcm and rate != self.decoded_rate:
                info += f"  →  will resample to {self.decoded_rate} Hz"
            elif not self.decoded_pcm:
                info += "  (will resample to XA track rate on replace)"
            self.l_wav_info.configure(text=info)
            self._log(f"WAV: {os.path.basename(p)}  "
                      f"{rate} Hz  {'stereo' if ch==2 else 'mono'}  "
                      f"{bits}-bit  {dur:.2f}s")
            if self.decoded_pcm and rate != self.decoded_rate:
                self._log(
                    f"  ℹ Will resample {rate} Hz → {self.decoded_rate} Hz on replace")
            self.b_prv.configure(state='normal')
            self._upd_btns()
        except Exception as e:
            messagebox.showerror("WAV Error", str(e))

    def do_replace(self, overwrite: bool = False):
        if not self.sel_track or not self.wav_pcm:
            messagebox.showwarning("Missing data",
                                   "Select a track and load a WAV file first.")
            return
        if overwrite:
            if not self.source_path:
                messagebox.showerror("Error", "No source file path available.")
                return
            if not messagebox.askyesno(
                    "Overwrite original file?",
                    f"This will OVERWRITE:\n\n{self.source_path}\n\n"
                    f"Only track {self.sel_track['index']+1}'s audio payload bytes "
                    f"will change.  All headers, other tracks, and non-audio sectors "
                    f"are untouched.\n\nProceed?",
                    icon='warning'):
                return
            out = self.source_path
        else:
            ext = '.bin' if self.fmt == 'full' else '.xa'
            out = filedialog.asksaveasfilename(
                title="Save modified BIN / XA",
                defaultextension=ext,
                filetypes=[("BIN image", "*.bin"),
                           ("XA audio",  "*.xa"),
                           ("All files", "*.*")])
            if not out: return

        self._busy(True)
        track = self.sel_track
        xc    = self.v_ch.get()
        xr    = self.v_rate.get()

        def worker():
            try:
                stereo          = (xc == 2)
                expected_stereo = bool(track['coding'] & COD_STEREO)
                expected_rate   = 18900 if (track['coding'] & COD_18900) else 37800
                if stereo != expected_stereo or xr != expected_rate:
                    self.after(0, lambda: self._log(
                        f"  ⚠ Override ({xc}ch / {xr} Hz) differs from "
                        f"native coding byte 0x{track['coding']:02X} "
                        f"({'stereo' if expected_stereo else 'mono'} / "
                        f"{expected_rate} Hz).  "
                        f"The subheader will be patched to match your override."))
                self.after(0, lambda: self._log(
                    f"Encoding replacement audio → XA ADPCM  "
                    f"(track {track['index']+1})…"))
                pcm = to_channels(self.wav_pcm, self.wav_ch, xc)
                if self.wav_rate != xr:
                    self.after(0, lambda: self._log(
                        f"  Resampling {self.wav_rate} Hz → {xr} Hz…"))
                    pcm = resample(pcm, self.wav_rate, xr, xc)
                blks  = encode_xa(pcm, stereo)
                n_xa  = track['n_sectors']
                n_blk = len(blks)
                self.after(0, lambda: self._log(
                    f"  XA sectors: {n_xa}  |  Encoded blocks: {n_blk}"))
                if n_blk < n_xa:
                    gap = n_xa - n_blk
                    blks += [bytes(AUDIO_BYTES)] * gap
                    self.after(0, lambda: self._log(
                        f"  ⚠ WAV shorter — {gap} sector(s) padded with silence"))
                elif n_blk > n_xa:
                    blks = blks[:n_xa]
                    self.after(0, lambda: self._log(
                        f"  ⚠ WAV longer — truncated to {n_xa} sector(s)"))
                effective_coding = track['coding']
                if stereo != expected_stereo:
                    effective_coding = (effective_coding | COD_STEREO
                                        if stereo else effective_coding & ~COD_STEREO)
                if xr != expected_rate:
                    effective_coding = (effective_coding | COD_18900
                                        if xr == 18900 else effective_coding & ~COD_18900)
                effective_track = dict(track, coding=effective_coding)
                if overwrite:
                    overwrite_track(out, effective_track, blks, self.fmt)
                    self.source_raw = rebuild_bin(
                        self.source_raw, effective_track, blks, self.fmt)
                else:
                    result = rebuild_bin(self.source_raw, effective_track, blks, self.fmt)
                    with open(out, 'wb') as f:
                        f.write(result)
                self.after(0, lambda: self._log(f"  ✔ Saved: {out}"))
                self.after(0, lambda: messagebox.showinfo(
                    "Done",
                    f"Track {track['index']+1} replaced successfully.\n\n{out}"))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
                self.after(0, lambda: self._log(f"  ✗ {e}"))
            finally:
                self.after(0, lambda: self._busy(False))

        threading.Thread(target=worker, daemon=True).start()

    def _add_to_queue(self):
        if not self.sel_track or not self.wav_pcm:
            messagebox.showwarning("Missing data",
                                   "Select a track and load a WAV file first.")
            return
        track = self.sel_track
        iid   = str(self._bulk_iid_counter)
        self._bulk_iid_counter += 1
        entry = {
            'iid'     : iid,
            'track'   : track,
            'wav_pcm' : list(self.wav_pcm),
            'wav_ch'  : self.wav_ch,
            'wav_rate': self.wav_rate,
            'wav_name': self.l_wav.cget('text'),
            'xc'      : self.v_ch.get(),
            'xr'      : self.v_rate.get(),
        }
        self.bulk_queue.append(entry)
        self.bulk_tree.insert('', 'end', iid=iid, values=(
            track['index'] + 1,
            f"0x{track['file_num']:02X}",
            f"0x{track['channel_num']:02X}",
            'Stereo' if entry['xc'] == 2 else 'Mono',
            f"{entry['xr']} Hz",
            entry['wav_name'],
            'Pending',
        ))
        self._upd_bulk_btns()
        self._log(f"  + Queue [{len(self.bulk_queue)}]: "
                  f"Track {track['index']+1} → {entry['wav_name']}")

    def _remove_from_queue(self):
        sel = self.bulk_tree.selection()
        if not sel:
            return
        for iid in sel:
            self.bulk_queue = [e for e in self.bulk_queue if e['iid'] != iid]
            self.bulk_tree.delete(iid)
        self._upd_bulk_btns()

    def _clear_queue(self):
        self.bulk_queue.clear()
        for row in self.bulk_tree.get_children():
            self.bulk_tree.delete(row)
        self._upd_bulk_btns()
        self._log("  Bulk queue cleared.")

    def bulk_replace(self, overwrite: bool = False):
        if not self.bulk_queue:
            messagebox.showwarning("Empty Queue",
                                   "Add at least one track→WAV mapping first.")
            return
        if overwrite:
            if not self.source_path:
                messagebox.showerror("Error", "No source file path available.")
                return
            if not messagebox.askyesno(
                    "Overwrite original file?",
                    f"This will OVERWRITE:\n\n{self.source_path}\n\n"
                    f"Replacing {len(self.bulk_queue)} track(s).\n\nProceed?",
                    icon='warning'):
                return
            out = self.source_path
        else:
            ext = '.bin' if self.fmt == 'full' else '.xa'
            out = filedialog.asksaveasfilename(
                title="Save modified BIN / XA",
                defaultextension=ext,
                filetypes=[("BIN image", "*.bin"),
                           ("XA audio",  "*.xa"),
                           ("All files", "*.*")])
            if not out:
                return
        for b in (self.b_badd, self.b_brem, self.b_bclr, self.b_bsav, self.b_bovr):
            b.configure(state='disabled')
        self._busy(True)
        threading.Thread(
            target=self._do_bulk_replace,
            args=(overwrite, out),
            daemon=True).start()

    def _do_bulk_replace(self, overwrite: bool, out_path: str):
        try:
            total = len(self.bulk_queue)
            self.after(0, lambda: self._log(
                f"Bulk replace: {total} track(s) → {os.path.basename(out_path)}"))
            working_raw = self.source_raw
            for i, entry in enumerate(self.bulk_queue):
                track  = entry['track']
                xc     = entry['xc']
                xr     = entry['xr']
                stereo = (xc == 2)
                iid    = entry['iid']
                n      = i + 1
                self.after(0, lambda iid=iid: self.bulk_tree.set(
                    iid, 'Status', 'Encoding…'))
                msg = (f"  [{n}/{total}] Track {track['index']+1}  "
                       f"[0x{track['file_num']:02X} / "
                       f"0x{track['channel_num']:02X}]…")
                self.after(0, lambda m=msg: self._log(m))
                expected_stereo = bool(track['coding'] & COD_STEREO)
                expected_rate   = 18900 if (track['coding'] & COD_18900) else 37800
                if stereo != expected_stereo or xr != expected_rate:
                    warn = (f"    ⚠ Override ({xc}ch / {xr} Hz) differs from "
                            f"native 0x{track['coding']:02X} — subheader patched.")
                    self.after(0, lambda m=warn: self._log(m))
                try:
                    pcm = to_channels(entry['wav_pcm'], entry['wav_ch'], xc)
                    if entry['wav_rate'] != xr:
                        rs = f"    Resampling {entry['wav_rate']} Hz → {xr} Hz…"
                        self.after(0, lambda m=rs: self._log(m))
                        pcm = resample(pcm, entry['wav_rate'], xr, xc)
                    blks  = encode_xa(pcm, stereo)
                    n_xa  = track['n_sectors']
                    n_blk = len(blks)
                    if n_blk < n_xa:
                        gap  = n_xa - n_blk
                        blks += [bytes(AUDIO_BYTES)] * gap
                        pm = f"    ⚠ WAV shorter — {gap} sector(s) padded with silence"
                        self.after(0, lambda m=pm: self._log(m))
                    elif n_blk > n_xa:
                        blks = blks[:n_xa]
                        tm = f"    ⚠ WAV longer — truncated to {n_xa} sector(s)"
                        self.after(0, lambda m=tm: self._log(m))
                    effective_coding = track['coding']
                    if stereo != expected_stereo:
                        effective_coding = (effective_coding | COD_STEREO
                                            if stereo
                                            else effective_coding & ~COD_STEREO)
                    if xr != expected_rate:
                        effective_coding = (effective_coding | COD_18900
                                            if xr == 18900
                                            else effective_coding & ~COD_18900)
                    effective_track = dict(track, coding=effective_coding)
                    working_raw = rebuild_bin(
                        working_raw, effective_track, blks, self.fmt)
                    self.after(0, lambda iid=iid: self.bulk_tree.set(
                        iid, 'Status', 'Done ✔'))
                    dm = f"    ✔ Track {track['index']+1} encoded"
                    self.after(0, lambda m=dm: self._log(m))
                except Exception as e:
                    self.after(0, lambda iid=iid: self.bulk_tree.set(
                        iid, 'Status', 'Error ✗'))
                    raise
            with open(out_path, 'wb') as f:
                f.write(working_raw)
            if overwrite:
                self.source_raw = working_raw
            self.after(0, lambda: self._log(f"  ✔ Saved: {out_path}"))
            n_done = total
            self.after(0, lambda: messagebox.showinfo(
                "Done",
                f"{n_done} track(s) replaced successfully.\n\n{out_path}"))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Bulk Replace Error", str(e)))
            self.after(0, lambda: self._log(f"  ✗ {e}"))
        finally:
            self.after(0, lambda: self._busy(False))
            self.after(0, self._upd_bulk_btns)

if __name__ == '__main__':
    XATool().mainloop()
