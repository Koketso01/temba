import sys, yaml, numpy as np
from astropy.io import fits
F0, C = 1420.405751768, 299792.458
print('%-14s %8s %8s %8s   %s' % ('field', 'low 30', 'high 30', 'worst', 'rms per channel / median rms'))
for f in sys.argv[1:]:
    c = yaml.safe_load(open(f'fields/{f}/field.yaml'))
    with fits.open(c['substrate'], memmap=True) as hd:
        h, d = hd[0].header, hd[0].data.squeeze()
        nz, ny, nx = d.shape
        cy, cx, w = ny // 2, nx // 2, min(ny, nx) // 6
        rms = np.full(nz, np.nan)
        for k in range(nz):
            p = np.asarray(d[k, cy - w:cy + w:2, cx - w:cx + w:2], float)
            p = p[np.isfinite(p) & (p != 0)]
            if p.size > 100:
                rms[k] = 1.4826 * np.median(np.abs(p - np.median(p)))
    v = (float(h['CRVAL3']) + (np.arange(nz) + 1 - float(h['CRPIX3'])) * float(h['CDELT3'])) / 1e3
    fq = F0 * (1 - v / C)
    r = rms / np.nanmedian(rms)
    o = np.argsort(fq)
    lo, hi = np.nanmean(r[o[:30]]), np.nanmean(r[o[-30:]])
    k = int(np.nanargmax(r))
    print('%-14s %8.2f %8.2f %8.2f   worst at %.1f MHz (channel %d)' % (f, lo, hi, r[k], fq[k], k))
