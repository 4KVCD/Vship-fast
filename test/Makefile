.FORCE:

run:
	g++ libvshipTestIdentity.cpp -std=c++23 -g -lvship -o libvshipTestIdentity && ./libvshipTestIdentity

speedtest: .FORCE
	g++ speedtest.cpp -O3 -std=c++17 -g -lvship -o speedtest && ./speedtest -m ssimu2 --gpu-id 0 -g 4 -n 1000
