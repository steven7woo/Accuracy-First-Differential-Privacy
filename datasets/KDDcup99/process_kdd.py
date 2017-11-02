#!/usr/bin/python3

import sys, os
import math
import numpy as np
#import matplotlib.pyplot as plt

col_1_attrs = ['udp', 'tcp', 'icmp']
col_2_attrs = ['ssh', 'login', 'nntp', 'iso_tsap', 'pm_dump', 'Z39_50', 'eco_i', 'ftp', 'pop_3', 'rje', 'ctf', 'nnsp', 'courier', 'uucp', 'whois', 'daytime', 'smtp', 'kshell', 'sunrpc', 'uucp_path', 'telnet', 'vmnet', 'X11', 'discard', 'urp_i', 'shell', 'netstat', 'hostnames', 'http_443', 'http', 'ecr_i', 'auth', 'red_i', 'netbios_ssn', 'netbios_dgm', 'mtp', 'domain_u', 'printer', 'efs', 'time', 'urh_i', 'tim_i', 'other', 'ldap', 'domain', 'name', 'ftp_data', 'klogin', 'link', 'sql_net', 'private', 'pop_2', 'imap4', 'IRC', 'remote_job', 'exec', 'csnet_ns', 'tftp_u', 'bgp', 'echo', 'finger', 'ntp_u', 'supdup', 'systat', 'gopher', 'netbios_ns']
col_3_attrs = ['S1', 'S3', 'REJ', 'RSTOS0', 'OTH', 'SH', 'RSTR', 'SF', 'S2', 'S0', 'RSTO']
last_col_attrs = ['normal.', 'phf.', 'ipsweep.', 'spy.', 'pod.', 'smurf.', 'ftp_write.', 'satan.', 'land.', 'multihop.', 'imap.', 'teardrop.', 'neptune.', 'warezmaster.', 'warezclient.', 'loadmodule.', 'buffer_overflow.', 'nmap.', 'guess_passwd.', 'back.', 'perl.', 'rootkit.', 'portsweep.']


def log_transform(c):
  return math.log(1.0 + c)


X = []
Y = []

with open("kddcup.data_10_percent") as f:
  for line in f.readlines():
    words = line.strip().split(",")
    words2 = [words[0]] + words[4:-1] + ["-1" if words[-1]=="normal." else "1"]

    nums = list(map(float, words2))
    for ind in [0, 1, 2]:
      nums[ind] = log_transform(nums[ind])
    X.append(nums[:-1])
    Y.append(nums[-1])

X = np.array(X)
Y = np.array(Y)
x_norms = [sum(list(map(abs, x))) for x in X]
max_x_norm = max(x_norms)
y_norms = list(map(abs,Y))
max_y_norm = max(y_norms)

print("n = " + str(len(X)))
print("d = " + str(len(X[0])))
print("")
print("X norms:")
print("Max 1-norm: " + str(max_x_norm))
print("Average 1-norm: " + str(sum(x_norms)/len(x_norms)))
x_norms.sort()
print("Median 1-norm: " + str(x_norms[int(len(x_norms)/2)]))
print("")
print("Y norms:")
print("Max 1-norm: " + str(max_y_norm))
print("Average 1-norm: " + str(sum(y_norms)/len(y_norms)))
y_norms.sort()
print("Median 1-norm: " + str(y_norms[int(len(y_norms)/2)]))

#print("Maximum value of each column of X:")
#print([max(X[:,i]) for i in range(len(X[0]))])

#plt.figure()
#plt.plot(y_norms,np.linspace(0,1,len(y_norms)))
#plt.title("Y 1-norm CDF")
#plt.show()

with open("kddcup_dataset.txt", "w") as f:
  for x,y in zip(X,Y):
    xhat = [x/max_x_norm for x in x]
    yhat = y/max_y_norm
    f.write(" ".join(list(map(str, xhat))))
    f.write(" ")
    f.write(str(yhat))
    f.write("\n")

