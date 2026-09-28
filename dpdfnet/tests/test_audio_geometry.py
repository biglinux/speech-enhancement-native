"""Independent NumPy check of the specified audio framing, not execution of Rust."""
import numpy as np
import pytest

@pytest.mark.parametrize("length", [1, 479, 480, 481, 8191])
def test_pcm_framing_and_four_hop_dry_delay(length):
    n=960; hop=480; delay=6*hop
    rng=np.random.default_rng(5730)
    x=rng.normal(size=length)*0.1
    inp=np.pad(x,(0,delay+hop))
    window=np.sin(np.pi/2*np.sin(np.pi*(np.arange(n)+0.5)/n)**2)
    np.testing.assert_allclose(window[:hop]**2+window[hop:]**2,1,atol=1e-14)
    history=np.zeros(n);ola=np.zeros(n);pending=np.zeros(hop)
    spectra=np.zeros((5,n//2+1),np.complex128)
    result=np.zeros_like(inp)
    # Equivalent emit-old-pending / collect-hop scheduling, expressed at hop level.
    for pos in range(0,len(inp)-hop+1,hop):
        result[pos:pos+hop]=pending
        history[:hop]=history[hop:];history[hop:]=inp[pos:pos+hop]
        spectra[:-1]=spectra[1:];spectra[-1]=np.fft.rfft(history*window)
        ola+=np.fft.irfft(spectra[0],n=n)*window
        pending=ola[:hop].copy();ola[:hop]=ola[hop:];ola[hop:]=0
    np.testing.assert_allclose(result[:delay],0,atol=1e-14)
    np.testing.assert_allclose(result[delay:delay+length],x,atol=2e-14)
