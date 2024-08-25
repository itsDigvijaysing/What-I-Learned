Links: [IIT Hyderabad](IIT%20Hyderabad.md), [Computer Networks](../GATE%20Prep/Computer%20Networks.md)

Resources: [CN: Top Down Approach](https://gaia.cs.umass.edu/kurose_ross/online_lectures.htm)

# ACN - 01

- Language we are going to use C, [Cpp](../Cpp/Cpp.md), [Python](../Python/Python.md)
- Tools: Wireshark tool, NS-3 tool (Recommended Linux)
- TCP/IP Model on focus.
## Resources

- [Google Classroom]([CS5060: Advanced Computer Networks (Aug 2024) (google.com)](https://classroom.google.com/u/0/c/NjkxNjk0OTAyNDE0))
- [Slides of Lecture](https://drive.google.com/file/d/1CoNdo1VjZmc_0a0Q0sCCcin_x2lY8Ulx/view?usp=classroom_web&authuser=0)
## Syllabus

1. Basics of CN & Network
2. Application Layer
3. Transport Layer principles & protocols
4. Network Layer: Data Plane
5. Network Layer: Control Plane
6. Link Layer & LAN's

# ACN - 02

## Network Edge

### Component

- All **Devices:** such as 'Mobile, Camera, PC' that are connected at end of Network. 
- **Equipment:** Routers, switches, access points. 
- **Function:** Facilitates the entry and exit of data to and from the network, connecting end-users to the network core.
### Edge Computing

**Edge Computing** is a distributed computing paradigm where data processing and storage occur closer to the data source (Mini Processing Devices) or end-users rather than in a centralized data center. It can help to complete the small tasks which require real time changes, Later on that data can be sent to the cloud for later on usage.

**Key Points:**

- **Reduces Latency:** By processing data locally, it minimizes the delay caused by transmitting data to distant servers.
- **Improves Performance:** Enhances the speed and efficiency of applications, especially for real-time data processing.
- **Increases Reliability:** Reduces dependency on central servers, improving resilience to network outages.
- **Use Cases:** IoT devices, autonomous vehicles, smart cities, and remote monitoring systems.

## Packet Switching

**Packet Switching** is a method of data transmission where data is broken into small packets and sent independently over a network. 

**Key Points:**
- **Data Division:** Large messages are divided into smaller packets.
- **Routing:** Packets are sent via different paths and reassembled at the destination.
- **Efficiency:** Utilizes network resources more efficiently and handles variable traffic loads.
- **Reliability:** Ensures data delivery even if some packets are lost or delayed, with retransmission mechanisms.
### Types of Delays in Packet Switching

**Packet Delay** refers to the time it takes for a data packet to travel from its source to its destination in a network. It includes several components:

1. **Transmission Delay:** Time required to push the packet's bits into the transmission medium.
	- In the context of `packet delay`, **L / R** represents a simplified formula to calculate the transmission delay:
		- **L**: Packet Length (in bits)
		- **R**: Transmission Rate (in bits per second or bps)
			- **Transmission Delay Formula:**
				$$ Transmission Delay = \frac{L}{R} $$
				> This formula calculates the time it takes to push the entire packet of length **L** into the network link with a rate of **R** bits per second. For example, if a packet is 1,000,000 bits long and the transmission rate is 1,000,000 bps, the transmission delay would be 1 second.
				> 
2. **Propagation Delay:** Time for a packet to travel through the physical medium (e.g., cables, fiber optics).
3. **Queuing Delay:** Time a packet spends waiting in queue at network devices (e.g., routers, switches) before being processed.
4. **Processing Delay:** Time needed for routers or switches to process the packet's header and routing information.

Total packet delay is the sum of these delays and affects overall network performance and user experience.

## Services & Protocols of Internet

- `Protocol` define the format, order of message sent/received among network entities & action it tool on message transmission receipt
- **Duplex** refers to the mode of communication in a network:
	- **Simplex:** One-way communication (e.g., radio broadcasts).
	- **Half-Duplex:** Two-way communication, but not simultaneous (e.g., walkie-talkies).
	- **Full-Duplex:** Two-way communication, simultaneous (e.g., telephone conversations).

### **Internet Services:**

1. **Web Browsing:** Access and view web pages.
2. **Email:** Send and receive electronic mail.
3. **File Sharing:** Transfer files between systems.
4. **Domain Name Resolution:** Convert domain names to IP addresses.
5. **Remote Access:** Access and control systems from a distance.
6. **Streaming Media:** Deliver audio and video content in real-time.

### **Internet Protocols:**

1. **HTTP/HTTPS:** Protocols for web page transfer; HTTPS includes encryption.
2. **SMTP:** Protocol for sending emails.
3. **POP3/IMAP:** Protocols for retrieving and managing emails.
4. **FTP/SFTP:** Protocols for file transfer; SFTP includes encryption.
5. **DNS:** Protocol for resolving domain names to IP addresses.
6. **Telnet/SSH:** Protocols for remote system access; SSH is secure.
7. **RTSP/RTP:** Protocols for streaming media delivery.

## Physical Layer Devices Types:

1. **Twisted Pair Cables:**
   - **Types:** Unshielded Twisted Pair (UTP), Shielded Twisted Pair (STP).
   - **Usage:** Common in Ethernet networks.
   - **Example:** Cat5e, Cat6 cables.

2. **Coaxial Cables:**
   - **Usage:** Used for cable TV and older Ethernet networks.
   - **Example:** RG-6, RG-59 cables.

3. **Fiber Optic Cables:**
   - **Types:** Single-mode, Multi-mode.
   - **Usage:** High-speed, long-distance data transmission.
   - **Example:** OM3, OM4 cables for multi-mode; OS1, OS2 cables for single-mode.

4. **Wireless Media:**
   - **Types:** Radio Waves, Infrared.
   - **Usage:** Wi-Fi, Bluetooth, satellite communication.
   - **Example:** Wi-Fi routers, Bluetooth devices, satellite dishes.

5. **Serial and Parallel Ports:**
   - **Usage:** Older interfaces for connecting peripherals.
   - **Example:** RS-232 serial ports, IEEE 1284 parallel ports.

6. **Satellites:**
   - **Usage:** Provide communication over long distances, including global and remote areas.
   - **Example:** Communication satellites for TV broadcasting and Internet services.

7. **Wireless Radios:**
   - **Usage:** Facilitate wireless communication over short and long distances.
   - **Example:** Wi-Fi routers, cellular base stations.

## Shannon Capacity Theorem:

**Shannon's Theorem** (Shannon-Hartley Theorem) relates to the maximum data rate of a communication channel. It defines the theoretical upper limit of data transmission capacity, given the channel's bandwidth and signal-to-noise ratio.

**Formula:**

$$[C = B \log_2 \left(1 + \frac{S}{N}\right) ]$$

**Where:**
- **C** = Channel capacity (in bits per second)
- **B** = Bandwidth of the channel (in hertz)
- **S/N** = Signal-to-noise ratio (power ratio)

**Key Points:**
- **Channel Capacity (C):** Maximum data rate that can be transmitted over the channel with no errors.
- **Bandwidth (B):** Range of frequencies available for transmission.
- **Signal-to-Noise Ratio (S/N):** Ratio of signal power to noise power, affecting the clarity of the signal.

Shannon's Theorem provides the foundation for understanding and optimizing communication systems.

## Extra

- Physical Layer also use electromagnetic waves same as wireless network & Optical fiber uses light pulses.
- `Internet Standard` Set by:
	- RFC: Request for Comment
	- IETF: Internet Engineering Task Force

# ACN - 03

## Resources:
https://classroom.google.com/u/3/c/NjkxNjk0OTAyNDE0

## Physical Link Media:

![Physical Media](../Archive/Attachment/Physical%20Media.png)
- BW = f2 (Max Range) - f1 (Min Range)
## Access Network: Digital Subscriber Line (DSL)

- Telephone require point to point connection dedicated line to central office.
- In Below connection user can only use Either Internet or Telephone at one time. User can't use both at same time.
![DSL](../Archive/Attachment/DSL.png)
- ADSL : Async Digital Subscriber Line
	- One Line Holds Voice (B1), Up Link (B2), Down Link(B3)
	- Total Bandwidth B=B1+B2+B3. (B1= Voice require very less data(10 to 16 kbps) so we can also neglect it)
	- Generally Down link Bandwidth is set to be higher because majority or time we only download from internet.
- frequency division multiplexing (FDM) (Hybrid Cable): different TV channels & data transmitted in different frequency bands on the shared coaxial cable

- **Access Network Wireless/WiFi**: Provides wireless connectivity for devices, commonly used in homes, offices, and public spaces.
- **Access Network Homes (FTTH)**: Fiber-to-the-Home (FTTH) delivers high-speed internet via optical fiber directly to residences.
- **Access Network Enterprises (Wired/Wireless)**: Enterprises may use a mix of wired connections (Ethernet) for stability and wireless networks (WiFi/WLAN) for flexibility.
- **Access Network Data Centers**: Typically rely on high-speed, wired connections for data transfer and management.
- **Access Network Satellites**: Used for internet access in remote or rural areas, and for global communication networks, offering connectivity where traditional infrastructure is unavailable.

##  Network speed

**Ethernet:**
- **10BASE-T:** 10 Mbps
- **100BASE-TX:** 100 Mbps
- **1000BASE-T:** 1 Gbps
- **10GBASE-T:** 10 Gbps
- **25GBASE-T:** 25 Gbps
- **40GBASE-T:** 40 Gbps

**Wi-Fi:**
- **Wi-Fi 1 (802.11b):** Up to 11 Mbps
- **Wi-Fi 2 (802.11a):** Up to 54 Mbps
- **Wi-Fi 3 (802.11g):** Up to 54 Mbps
- **Wi-Fi 4 (802.11n):** Up to 600 Mbps
- **Wi-Fi 5 (802.11ac):** Up to 3.5 Gbps
- **Wi-Fi 6 (802.11ax):** Up to 9.6 Gbps
- **Wi-Fi 7 (802.11be):** Up to 30 Gbps

# ACN - 04

## Network Core:
- mesh of interconnected routers
- packet-switching: hosts break application-layer messages into packets
- network forwards packets from one router to the next, across links on path from source to destination
- Two main Function of Network Core are:
	- Forwarding: (Switching) Moving router packet to appropriate router point.
	- Routing: Determines source & Destination path which will be taken by packets.

![Packet Switching](../Archive/Attachment/ACN%20Packet%20Switching.png)

## Packet Switching
Packet Switching have two different types: 
> L = Length of Packet
> R = Speed of Transfer bits/sec
> D = Distance
> S = Speed of Light
1. Store & Forward Switching:
   Entire Packet must be travelled to next router before transferring the next packet, it will wait till complete packet transfer. Solved E.g. in Notebook.
   $$\frac{L}{R}+\frac{D}{S}$$
2. Cut-Through Switching (Pass Through): 
   Router starts transmitting to destination & do not wait to receive complete packet.
   $$\frac{1}{R}+\frac{D}{S}$$
   > L=1, because it send 1-bit without waiting that's why it's fast, but it's used in very specific networks (Enterprises), because there can be in between packet loss but will not be verified & it require network to work at similar speed because if not switch will struggle to handle the data.
   
   ![Store and Forward working](../Archive/Attachment/Store%20and%20Forward%20working.png)
## Circuit Switching

**Definition**: Circuit switching is a method of communication where a dedicated communication path or circuit is established between the source and the destination for the duration of the communication session.

**Key Features**:
- **Dedicated Path**: A fixed path is reserved exclusively for the entire communication session.
- **Phases**:
  1. **Setup Phase**: The circuit is established before data transmission begins.
  2. **Data Transfer Phase**: Data is transmitted through the reserved circuit.
  3. **Teardown Phase**: The circuit is released once the communication is complete.
- **Example**: Traditional telephone networks, where a call sets up a dedicated line between the caller and the receiver.

**Advantages**:
- **Guaranteed Bandwidth**: The circuit provides a consistent and guaranteed bandwidth.
- **Low Latency**: Minimal delay once the circuit is established.

**Disadvantages**:
- **Inefficient Use of Resources**: The dedicated path remains reserved even if no data is being transmitted, leading to potential wastage.
- **Setup Time**: Time is required to establish the circuit before data transmission can begin.

# ACN - 05

## Packet Switching vs. Circuit Switching

#### **Packet Switching**
- **How it Works**: Data is broken into small packets that are sent independently over the network. Each packet can take a different path to the destination, where they are reassembled.
- **Flexibility**: Efficient use of network resources since the same paths can be shared by multiple connections.
- **Example**: The Internet (e.g., emails, web browsing).

**Benefits:**
- **Efficient Resource Utilization**: Multiple users can share the same network paths, making better use of available bandwidth.
- **Scalability**: Easily accommodates a large number of users and data transmissions.
- **Robustness**: If one path fails, packets can be rerouted through alternative paths, making the network more resilient to failures.
- **Cost-Effective**: No need for dedicated lines, reducing infrastructure costs.

**Disadvantages:**
- **Potential for Delay**: Packets can take different routes and may arrive out of order, causing potential delays and the need for reassembly.
- **Variable Latency**: Since packets may follow different paths, the time they take to reach the destination can vary.
- **Complexity**: Requires sophisticated protocols to handle packet routing, reassembly, and error checking.

#### **Circuit Switching**
- **How it Works**: A dedicated communication path (circuit) is established between the source and destination for the entire duration of the communication session.
- **Reliability**: Provides consistent and reliable communication with guaranteed bandwidth, but can be inefficient since the circuit is reserved even if no data is being sent.
- **Example**: Traditional telephone networks.

**Benefits:**
- **Consistent Performance**: Provides a guaranteed, dedicated bandwidth with predictable latency, making it ideal for real-time communication (e.g., voice calls).
- **Reliability**: Once the circuit is established, the connection is stable and secure throughout the communication session.
- **Low Latency**: Since the path is dedicated, data is transmitted without delays or interruptions.

**Disadvantages:**
- **Inefficient Resource Utilization**: The dedicated circuit remains reserved even when no data is being transmitted, leading to potential wastage of network resources.
- **Setup Time**: Establishing the circuit can take time, leading to delays before communication begins.
- **Cost**: Requires dedicated infrastructure and maintenance, which can be expensive compared to packet-switched networks.

## Circuit Switching Types:

- Frequency Division Multiplexing & Time Division Multiplexing
![FDM & TDM](../Archive/Attachment/FDM%20&%20TDM.png)

### Problem & Solution:

![Circuit & Packet Switching](../Archive/Attachment/ACN%20Circuit%20&%20Packet%20switching.png)
- It shows that in packet Switching performs much better then circuit switching in such cases.
### Binomial Distribution Overview

**Definition**: The binomial distribution models the number of successes in a fixed number of independent trials, where each trial has two possible outcomes: success (with probability p) and failure (with probability 1−p).

**Key Parameters**:

- n: Number of trials.
- P: Probability of success on each trial.
- r: Random variable representing the number of successes in n trials.

**Probability Mass Function (PMF)**:

$$P(X=r)=(^nCr)p^r(1−p)^{(n−r)}$$

Where:

- (nCr) is the binomial coefficient, representing the number of ways to choose r successes from n trials.
- r is the number of successes (where 0≤r≤n).

## Why Packet Sharing:

- Resource Sharing so High Users can use simultaneously.
- Simpler and No High Presetup.
- Excessive Congestion Possible (Packet Delay, Packet loss due to buffer overflow, so need protocols for reliability(Congestion Control))

# ACN - 06

## 1. Internet Structure: Network of Networks

- The Internet is a **"network of networks"** where multiple networks interconnect.
- Each network is owned and operated by different organizations (e.g., ISPs, enterprises, universities).
- Networks are organized hierarchically, with access ISPs, regional ISPs, and global transit ISPs.
- **Hybrid Topologies** can be created by connecting access ISPs to global transit ISPs.

### Example of Hybrid Topologies
- **Hybrid Topology**: A combination of different network topologies (e.g., star, mesh, bus) to optimize performance and scalability.
  - **Example**: Connecting each Access ISP to a Global Transit ISP.

## 2. Network Typologies (Network Structure)

- **Network Topology**: The structure/layout of how nodes (e.g., computers, routers) are interconnected in a network.
  - **Types**:
    - **Bus**: All devices share a common communication line.
    - **Star**: All devices are connected to a central hub.
    - **Ring**: Devices are connected in a circular fashion.
    - **Mesh**: Devices are interconnected, providing multiple paths for data.
    - **Hybrid**: A combination of two or more topologies.

## 3. Packet Queue: Delay in Packets

- **Packet Queue**: Packets waiting in line to be processed by a router or network device.
  - **Delay** occurs when packets are waiting due to **congestion** in the queue.
  - **Types of Delays**:
    - **Processing Delay (d_process)**: Time to examine the packet header and determine where to direct the packet.
    - **Queueing Delay (d_queue)**: Time the packet spends waiting in the queue before it can be processed.
    - **Transmission Delay (d_tran)**: Time required to push all of the packet's bits onto the wire.
    - **Propagation Delay (d_prop)**: Time it takes for the signal to propagate from one end of the medium to the other.

### Total Delay (d_total)
\[
d_{\text{total}} = d_{\text{process}} + d_{\text{queue}} + d_{\text{tran}} + d_{\text{prop}}
\]

## 4. Total Loss Occurrence

- **Total Loss Occurrence**: The rate at which packets are lost due to congestion.
  - **Formula**: 
    \[
    \frac{L \cdot a}{R} 
    \]
    - **L**: Number of bits in the packet.
    - **a**: Arrival rate of bits to the queue (bits/sec).
    - **R**: Service rate of bits by the router or network device (bits/sec).
