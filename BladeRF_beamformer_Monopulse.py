"""
Jon Kraft, Nov 5 2022
https://github.com/jonkraft/Pluto_Beamformer
video walkthrough of this at:  https://www.youtube.com/@jonkraft

"""
# Copyright (C) 2020 Analog Devices, Inc.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#     - Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     - Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and/or other materials provided with the
#       distribution.
#     - Neither the name of Analog Devices, Inc. nor the names of its
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#     - The use of this software may or may not infringe the patent rights
#       of one or more patent holders.  This license does not release you
#       from the requirement that you obtain separate licenses from these
#       patent holders to use this software.
#     - Use of the software either in source or binary form, must be run
#       on or directly connected to an Analog Devices Inc. component.
#
# THIS SOFTWARE IS PROVIDED BY ANALOG DEVICES "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED.
#
# IN NO EVENT SHALL ANALOG DEVICES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, INTELLECTUAL PROPERTY
# RIGHTS, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
# THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Copyright (C) 2020 Analog Devices, Inc.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#     - Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     - Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and/or other materials provided with the
#       distribution.
#     - Neither the name of Analog Devices, Inc. nor the names of its
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#     - The use of this software may or may not infringe the patent rights
#       of one or more patent holders.  This license does not release you
#       from the requirement that you obtain separate licenses from these
#       patent holders to use this software.
#     - Use of the software either in source or binary form, must be run
#       on or directly connected to an Analog Devices Inc. component.
#
# THIS SOFTWARE IS PROVIDED BY ANALOG DEVICES "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED.
#
# IN NO EVENT SHALL ANALOG DEVICES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, INTELLECTUAL PROPERTY
# RIGHTS, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
# THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import adi
import matplotlib.pyplot as plt
import numpy as np
from bladerf import _bladerf as bladeRF
from matplotlib.animation import FuncAnimation


'''Setup'''
samp_rate = 2e6    # must be <=30.72 MHz if both channels are enabled
num_samples = 2**12
rx_lo = 5.5e9
fc0 = int(200e3)
phase_cal = 0

''' Set distance between Rx antennas '''
d_wavelength = 0.5                  # distance between elements as a fraction of wavelength.  This is normally 0.5
wavelength = 3E8/rx_lo              # wavelength of the RF carrier
d = d_wavelength*wavelength         # distance between elements in meters
print("Set distance between Rx Antennas to ", int(d*1000), "mm")

devices = bladeRF.get_device_list()
bladerf = bladeRF.BladeRF(devinfo=devices[0])

for chN in [0, 1]:
    ch = bladerf.Channel(bladeRF.CHANNEL_RX(chN))
    ch.sample_rate = samp_rate
    ch.bandwidth = int(fc0*3)
    ch.gain = 0
    ch.frequency = rx_lo
    ch.enable = True

sync_config = dict(
    layout          = bladeRF.ChannelLayout.RX_X2,
    fmt             = bladeRF.Format.SC16_Q11,
    num_buffers     = 32,
    buffer_size     = 32768,
    num_transfers   = 16,
    stream_timeout  = 1000
)

def rx():
    # function to receive IQ samples 
    bladerf.sync_config(**sync_config)

    bytes_per_sample = 8
    nb_each_16bits = 2
    num_channel = 2

    buf = bytearray(2*num_samples*num_channel*nb_each_16bits)
    num = min(len(buf)//bytes_per_sample,num_samples)
    # Read into buffer
    bladerf.sync_rx(buf, num)
    data = np.frombuffer(buf, dtype=np.int16)
    # Розділення каналів
    iq_ch0 = data[0::4] + 1j * data[1::4]  # Сигнали RX0
    iq_ch2 = data[2::4] + 1j * data[3::4]  # Сигнали RX2
    return iq_ch0, iq_ch2

'''Program Tx and Send Data'''
fs = int(samp_rate)
N = 2**16
ts = 1 / float(fs)
t = np.arange(0, N * ts, ts)
i0 = np.cos(2 * np.pi * t * fc0) * 2 ** 14
q0 = np.sin(2 * np.pi * t * fc0) * 2 ** 14
iq0 = i0 + 1j * q0

# Assign frequency bins and "zoom in" to the fc0 signal on those frequency bins
xf = np.fft.fftfreq(num_samples, ts)
xf = np.fft.fftshift(xf)/1e6
signal_start = int(num_samples*(samp_rate/2+fc0/2)/samp_rate)
signal_end = int(num_samples*(samp_rate/2+fc0*2)/samp_rate)

def calcTheta(phase):
    # calculates the steering angle for a given phase delta (phase is in deg)
    # steering angle is theta = arcsin(c*deltaphase/(2*pi*f*d)
    arcsin_arg = np.deg2rad(phase)*3E8/(2*np.pi*rx_lo*d)
    arcsin_arg = max(min(1, arcsin_arg), -1)     # arcsin argument must be between 1 and -1, or numpy will throw a warning
    calc_theta = np.rad2deg(np.arcsin(arcsin_arg))
    return calc_theta

def dbfs(raw_data):
    # function to convert IQ samples to FFT plot, scaled in dBFS
    num_samples = len(raw_data)
    win = np.hamming(num_samples)
    y = raw_data * win
    s_fft = np.fft.fft(y) / np.sum(win)
    s_shift = np.fft.fftshift(s_fft)
    s_dbfs = 20*np.log10(np.abs(s_shift)/(2**11))     # Pluto is a signed 12 bit ADC, so use 2^11 to convert to dBFS
    return s_shift, s_dbfs

def monopulse_angle(array1, array2):
    ''' Correlate the sum and delta signals  '''
    # Since our signals are closely aligned in time, we can just return the 'valid' case where the signals completley overlap
    # We can do correlation in the time domain (probably faster) or the freq domain
    # In the time domain, it would just be this:
    # sum_delta_correlation = np.correlate(delayed_sum, delayed_delta, 'valid')
    # But I like the freq domain, because then I can focus just on the fc0 signal of interest
    sum_delta_correlation = np.correlate(array1[signal_start:signal_end], array2[signal_start:signal_end], 'valid')
    angle_diff = np.angle(sum_delta_correlation)
    return angle_diff

def scan_for_DOA():
    # go through all the possible phase shifts and find the peak, that will be the DOA (direction of arrival) aka steer_angle
    data = rx()
    Rx_0=data[0]
    Rx_1=data[1]
    peak_sum = []
    peak_delta = []
    monopulse_phase = []
    delay_phases = np.arange(-180, 180, 2)    # phase delay in degrees
    for phase_delay in delay_phases:   
        delayed_Rx_1 = Rx_1 * np.exp(1j*np.deg2rad(phase_delay+phase_cal))
        delayed_sum = Rx_0 + delayed_Rx_1
        delayed_delta = Rx_0 - delayed_Rx_1
        delayed_sum_fft, delayed_sum_dbfs = dbfs(delayed_sum)
        delayed_delta_fft, delayed_delta_dbfs = dbfs(delayed_delta)
        mono_angle = monopulse_angle(delayed_sum_fft, delayed_delta_fft)
        
        peak_sum.append(np.max(delayed_sum_dbfs))
        peak_delta.append(np.max(delayed_delta_dbfs))
        monopulse_phase.append(np.sign(mono_angle))
        
    peak_dbfs = np.max(peak_sum)
    peak_delay_index = np.where(peak_sum==peak_dbfs)
    peak_delay = delay_phases[peak_delay_index[0][0]]
    steer_angle = int(calcTheta(peak_delay))
    
    return delay_phases, peak_dbfs, peak_delay, steer_angle, peak_sum, peak_delta, monopulse_phase  
    

fig, ax = plt.subplots()
plt.ylim(top=5, bottom=-30)        
plt.xlabel("phase shift [deg]")
plt.ylabel("Rx0 + Rx1 [dBfs]")
ln, = plt.plot([], [], 'ro')


def animate(ival):
    ax.clear()
    delay_phases, peak_dbfs, peak_delay, steer_angle, peak_sum, peak_delta, monopulse_phase = scan_for_DOA()
    plt.plot(delay_phases, peak_sum)
    plt.plot(delay_phases, peak_delta)
    plt.plot(delay_phases, monopulse_phase)
    plt.axvline(x=peak_delay, color='r', linestyle=':')
    plt.text(-180, -26, "Peak signal occurs with phase shift = {} deg".format(round(peak_delay,1)))
    plt.text(-180, -28, "If d={}mm, then steering angle = {} deg".format(int(d*1000), steer_angle))
    plt.draw()
    return ln,

ani = FuncAnimation(fig, animate, frames=1000, interval=0)
plt.show()

    

    

        
   
