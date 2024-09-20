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

![ISP](../Archive/Attachment/ISP.png)

### Example of Hybrid Topologies
- **Hybrid Topology**: A combination of different network topologies (e.g., star, mesh, bus) to optimize performance and scalability.
  - **Example**: Connecting each Access ISP to a Global Transit ISP.

## 2. Network Typologies (Network Structure)

- **Network Topology**: The structure/layout of how nodes (e.g., computers, routers) are interconnected in a network.
![Network Topologies](../Archive/Attachment/Network%20Topologies.png)
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
$$[
d_{\text{total}} = d_{\text{process}} + d_{\text{queue}} + d_{\text{tran}} + d_{\text{prop}}
]$$

## 4. Total Loss Occurrence

- **Total Loss Occurrence**: The rate at which packets are lost due to congestion.
  - **Formula**: 
    $$
    \frac{L \cdot a}{R} 
    $$
    - **L**: Number of bits in the packet.
    - **a**: Arrival rate of bits to the queue (bits/sec).
    - **R**: Service rate of bits by the router or network device (bits/sec).

#### Service Requirements
![Service Requirements](../Archive/Attachment/Service%20Requirements.png)

# ACN - 07

## Packet Queue Delay

![Packet delay](../Archive/Attachment/Packet%20Delay.png)
### Scenario 1: Packets Arrive Simultaneously
- **Time (t = t1):** All 5 packets arrive at the same time (Inter-Arrival Time, IAT = 0).
- **Consequence:** Every subsequent packet has to wait for the previous ones to be transmitted.
  - **Wait Time:** Each packet after the first has to wait ((n-1){L} / {R}) for its turn to be transmitted.
  $$\frac{[n-1]L}{R}$$

### Scenario 2: Packets Arrive Late
- **Time (t = t2):** All 5 packets arrive with a delay (IAT > Transmission Delay).
- **Consequence:** Depending on the delay, packets may experience reduced waiting time or none at all if the link is idle.

### Queue Dynamics
- If packets queue in router buffers, they wait their turn for transmission.
  - **Queue Length Growth:** Occurs when the arrival rate to the link temporarily exceeds the output link capacity.
  - **Packet Loss:** Happens when the memory allocated for queued packets is full, leading to packet drops.

## Real Internet Delay & Routes

### Traceroute Program
- **Function:** Provides delay measurement from the source to the destination.
- **Types of Packets Supported:** TCP, UDP, ICMP.
- **Time to Live (TTL):** Default is set to 64.

### Packet Loss Scenarios
- **TTL Expiration:**
  - **Option 1:** Router replies by creating a "Time Exceeded" error; the result is visible as `***`.
  - **Option 2:** Router does not reply if set by the router's policy.

- **Buffer Size Limit Exceeded:**
  - **Cause:** Packet loss can occur if the buffer size limit is exceeded, leading to a full queue.
  - **Consequence:** Dropped or lost packets during transmission.

### Key Points
- Packet delays can occur due to congestion in queues.
- Packet loss can be caused by buffer overflow or TTL expiration.
- Traceroute helps diagnose delay and packet loss issues in a network.

# ACN - 08

## Throughput

- **Throughput**: The rate at which bits are being sent from the sender to the receiver.
  - **Instantaneous Throughput**: The rate at a given point in time.
  - **Average Throughput**: The rate over a longer period of time.

### Throughput Calculation
- **Formula**:
$$[
  \text{Throughput (Tput)} = \frac{\text{Message Size (M)}}{\text{Message Delay (D)}} = \frac{M}{\text{Transmission Delay} + \text{Propagation Delay}}
  ]$$
- **Result**:
$$  [
  \text{Tput} = R_c \text{ (Rate after calculation)}
  ]$$
  
- **Bottlenecks**: The throughput is often limited by the slower of the two rates:
![Through put](../Archive/Attachment/Through%20put.png)
  - **R_client**: Rate of the client.
  - **R_server**: Rate of the server.

## Delay × Bandwidth Product

- **Concept**: The link between a pair of nodes can be visualized as a hollow pipe.
  - **Latency**: The length of the pipe (time delay).
  - **Bandwidth**: The width of the pipe (amount of data that can be transmitted per unit time).
![Delay X Bandwidth](../Archive/Attachment/Delay%20X%20Bandwidth.png)

### Key Points
- **Delay × Bandwidth Product**: 
  - **Meaning**: It indicates how many bits the sender must transmit before the first bit arrives at the receiver if the sender wants to keep the pipe full.
  - **Response Time**: Takes another one-way latency to receive a response (ACK) from the receiver.
  - **Utilization**: If the sender does not send a full delay × bandwidth product’s worth of data before waiting for an ACK, the network’s capacity will not be fully utilized.

### Layering TCP

![TCP Layering](../Archive/Attachment/TCP%20Layering.png)

# ACN - 09

## Services, Layering, and Encapsulation in TCP/IP

### 1. Application Layer
- **Function**: The application layer exchanges messages to implement a specific application service.
- **Interaction**: Uses the services provided by the transport layer to facilitate communication.

### 2. Transport Layer
- **Function**: Transfers the message (M) from one process to another.
  - **Reliability**: Ensures reliable transfer of data (e.g., TCP).
- **Interaction**: Uses the services provided by the network layer to move data between processes on different hosts.

### 3. Network Layer
- **Function**: Transfers the transport-layer segment \([Ht | M]\) from one host to another.
- **Interaction**: Utilizes the link layer services to facilitate host-to-host communication.

### 4. Link Layer
- **Function**: Transfers the datagram \([Hn| [Ht |M]]\) from a host to a neighboring router.
- **Interaction**: Uses the services of the physical layer to transmit data between devices.

### 5. Physical Layer
- **Function**: Provides the physical means of transmitting raw bits over a communication link.

![ACN Pic](../Archive/Attachment/ACN%20Pic.png)

## Encapsulation

- **Concept**: Each layer adds its own header information (H) to the data received from the layer above before passing it to the layer below.
- **Process**:
  1. **Message (M)**: Application data.
  2. **Segment**: Transport layer encapsulates the message into a segment with a transport header ([Ht | M]).
  3. **Datagram**: Network layer encapsulates the segment into a datagram with a network header ([Hn| [Ht | M]]).
  4. **Frame**: Link layer encapsulates the datagram into a frame for transmission.
![Packet Transfer](../Archive/Attachment/Packet%20Transfer.png)
- **Matryoshka Dolls**: Encapsulation is like stacking Matryoshka dolls, where each layer adds its own information, wrapping the previous layer's data.

### Summary of Encapsulation Terms
- **Message**: Data at the application layer.
- **Segment**: Data at the transport layer.
- **Datagram**: Data at the network layer.
- **Frame**: Data at the link layer.

# ACN - 10

## What is a Socket?

- A **socket** is an API (Application Programming Interface) that allows networking applications to use the services of the transport layer.
- It acts like a pipe, providing communication between the transport layers of two hosts within a network.
![Socket](../Archive/Attachment/Socket.png)
### Key Characteristics of Sockets:
- **API between application and transport layers:**
  - **Berkeley sockets interface**: Originally provided by BSD 4.1 around 1982.
- **Socket Operations:**
  - **Creating a socket**: Initialize a socket for communication.
  - **Binding the socket**: Attach the socket to a network (e.g., bind to an IP address and port).
  - **Sending/receiving data**: Use the socket to transmit or receive data between hosts.
  - **Closing the socket**: Properly terminate the communication by closing the socket.

## Socket Programming

### Goal:
- The primary goal of socket programming is to learn how to build client/server applications that communicate using sockets.

### Definition:
- A **socket** is a door or interface between an application process and the end-to-end transport protocol (TCP/UDP).

### Types of Sockets:
- **UDP Sockets**: 
  - Unreliable, datagram-based communication.
- **TCP Sockets**: 
  - Reliable, byte stream-oriented communication.

## Application Example

### Example: Simple Client/Server Communication

1. **Client**:
   - Reads a line of characters (data) from its keyboard.
   - Sends the data to the server.
   
2. **Server**:
   - Receives the data from the client.
   - Converts the characters to uppercase.
   - Sends the modified data back to the client.
   
3. **Client**:
   - Receives the modified data from the server.
   - Displays the modified line on its screen.

### Example in Python:

#### Server Code:
```python
import socket

# Create a TCP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to an IP address and port
server_socket.bind(('localhost', 12345))

# Listen for incoming connections
server_socket.listen(1)
print("Server is listening...")

# Accept a connection
connection, address = server_socket.accept()
print(f"Connection from {address} has been established!")

# Receive data from the client
data = connection.recv(1024).decode('utf-8')
print(f"Received data: {data}")

# Convert data to uppercase
modified_data = data.upper()

# Send modified data back to the client
connection.send(modified_data.encode('utf-8'))

# Close the connection
connection.close()
```

#### Client Code:
```python
import socket

# Create a TCP socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the server
client_socket.connect(('localhost', 12345))

# Read a line of data from the keyboard
data = input("Enter a line of text: ")

# Send the data to the server
client_socket.send(data.encode('utf-8'))

# Receive the modified data from the server
modified_data = client_socket.recv(1024).decode('utf-8')
print(f"Modified data from server: {modified_data}")

# Close the socket
client_socket.close()
```

### Example in C++:

#### Server Code:
```cpp
#include <iostream>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

int main() {
    int server_socket = socket(AF_INET, SOCK_STREAM, 0);

    sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(12345);
    server_addr.sin_addr.s_addr = INADDR_ANY;

    bind(server_socket, (sockaddr*)&server_addr, sizeof(server_addr));

    listen(server_socket, 1);
    std::cout << "Server is listening..." << std::endl;

    int client_socket = accept(server_socket, nullptr, nullptr);
    
    char buffer[1024];
    memset(buffer, 0, sizeof(buffer));
    
    read(client_socket, buffer, sizeof(buffer));
    std::cout << "Received data: " << buffer << std::endl;

    // Convert to uppercase
    for(int i = 0; buffer[i]; i++) {
        buffer[i] = toupper(buffer[i]);
    }

    send(client_socket, buffer, strlen(buffer), 0);

    close(client_socket);
    close(server_socket);

    return 0;
}
```

#### Client Code:
```cpp
#include <iostream>
#include <cstring>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

int main() {
    int client_socket = socket(AF_INET, SOCK_STREAM, 0);

    sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(12345);
    inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr);

    connect(client_socket, (sockaddr*)&server_addr, sizeof(server_addr));

    std::string data;
    std::cout << "Enter a line of text: ";
    std::getline(std::cin, data);

    send(client_socket, data.c_str(), data.size(), 0);

    char buffer[1024];
    memset(buffer, 0, sizeof(buffer));

    read(client_socket, buffer, sizeof(buffer));
    std::cout << "Modified data from server: " << buffer << std::endl;

    close(client_socket);

    return 0;
}
```

# ACN - 11

## 1. Persistent HTTP

### Overview
- **Persistent HTTP** is the modern version of HTTP that allows the same TCP connection to be reused for multiple requests and responses.
- Earlier HTTP versions (like HTTP/1.0) closed the connection after each request, requiring a new TCP connection for every resource.

### Benefits of Persistent HTTP:
- **Reduced Connection Overhead**: Avoids the need to establish a new TCP connection for every file (HTML, images, etc.).
- **Improved Latency**: Since a single connection is used, round-trip times (RTTs) for connection setup are reduced.

### Example Problem:
- A webpage requires **1 HTML file** and **10 image files** to load.
- **No Pipelining (Sequential Requests)**:
  - Each file is requested one by one over a persistent connection, leading to multiple RTTs (request, response cycle).
- **With Pipelining**:
  - Multiple requests are sent without waiting for responses, reducing the delay.

## 2. Pipelining vs Non-pipelining (TCP Delays)

### Non-pipelined Requests (No Pipelining):
- Requests are made sequentially.
- Each subsequent request waits for the previous request's response, leading to high latency, especially over slow or long-distance connections.

### Pipelined Requests (With Pipelining):
- All requests are sent at once (without waiting for previous responses).
- TCP processes the responses in order.
- **Benefits**: Reduced waiting time, faster resource loading, especially noticeable when many small resources (like images) are requested.

## 3. Parallel TCP Sessions

### Overview:
- **Parallel TCP Sessions**: To further improve performance, multiple TCP connections (typically 6) are opened to download different resources concurrently.
- Commonly used to fetch assets (images, CSS, JS) in parallel for faster webpage rendering.

### Benefits:
- Increased throughput, as the browser can request multiple resources at once, utilizing the bandwidth more effectively.

## 4. HTTP/2

### Overview:
- **HTTP/2** is a major revision to HTTP designed to improve web performance, speed, and efficiency.
- It addresses many of the inefficiencies found in HTTP/1.x.

### Key Benefits:
1. **Multiplexing**: Multiple requests and responses can be sent over a single TCP connection, and they do not need to wait for one another. This reduces head-of-line blocking.
2. **Binary Framing**: HTTP/2 uses a binary protocol instead of textual. It splits the communication into **frames**, which improves performance and flexibility.
3. **Header Compression**: HTTP/2 compresses headers, reducing redundant data transmission.
4. **Server Push**: Allows servers to send resources to the client before the client explicitly requests them.

### HTTP/2 Frames vs Packets:
- **Frames**: HTTP/2 communication is divided into small frames, which can be interleaved and prioritized, allowing for more efficient data transmission.
- **Packets**: In traditional HTTP/1.x, packets often carried the entire request or response, leading to inefficiencies.
  
### Benefits of Frames:
- **Improved Packet Loss Recovery**: Since HTTP/2 divides data into frames, if a packet containing a frame is lost, only the missing frame needs to be retransmitted. This results in faster recovery and fewer delays compared to HTTP/1.x, where the loss of a packet could block the entire stream.

## 5. Web Performance Enhancements

### Improvements with HTTP/2:
- **Faster Load Times**: Due to multiplexing, header compression, and server push, web pages load faster.
- **Reduced Latency**: The elimination of head-of-line blocking and frame-based data transfer significantly improves latency, especially over high-latency networks.
- **Better Bandwidth Utilization**: Parallel transfer of resources over a single connection maximizes bandwidth use.

### Web Performance Summary:
- **HTTP/1.x** struggled with high latency and inefficient resource requests.
- **HTTP/2** solves many of these issues, leading to improved overall web performance, especially for modern, resource-heavy websites.

# ACN - 12

## 1. HTTP 1.x Messages

### Types of HTTP Messages
- **Request**: Sent by the client to request resources.
- **Response**: Sent by the server in reply to the request.

### Common HTTP Methods
- **GET**: Retrieves data from the server (e.g., a webpage).
- **POST**: Sends data to the server (e.g., form submission).
- **HEAD**: Similar to GET, but only retrieves headers (no body).
- **PUT**: Updates or uploads a resource to the server.

## 2. HTTP Response Codes
- **200 OK**: The request succeeded.
- **301 Moved Permanently**: The requested resource has been moved to a new URL.
- **304 Not Modified**: Indicates the resource has not been modified since last requested.
- **...**: Other response codes indicate various statuses (e.g., 404 Not Found, 500 Internal Server Error).

## 3. HEX & ASCII Values
- **HEX**: Represents binary data in a base-16 format. Commonly used in programming and networking for clarity.
- **ASCII**: Character encoding standard representing text in computers.
  - Example: The ASCII value of 'A' is 65 in decimal, 41 in hexadecimal.

## 4. HTTP/2.0 Features

### Server Push
- The server can send multiple resources to the client after receiving an HTTP GET request.
- **Considerations**: If objects are large, it can lead to increased delays and resource wastage.

### Multiplexing
- Allows multiple requests and responses to be sent simultaneously over a single connection, reducing latency.
  
### HTTP/2.0 with TLS
- HTTP/2 is often used with TLS (Transport Layer Security) to ensure secure data transmission.
- Enhances performance and security for web communications.

# ACN - 13

## 1. Maintaining User/Server State: Cookies

### HTTP Statelessness
- HTTP is a stateless protocol, meaning it does not retain user state or information across requests.

### HTTP + Cookies = Stateful
- Cookies enable stateful interactions, allowing the server to remember user information.

### Uses of Cookies
- **Authorization**: Verify user identity and grant access to secure areas.
- **Recommendations**: Suggest products or content based on user behavior.
- **Session Management**: Maintain user details during browsing sessions.
- **Shopping Carts**: Track items added to the cart across sessions.

## 2. Advertising Mechanisms

### Real-Time Bidding
- Advertisements on websites often use real-time bidding to display relevant ads.
- Hundreds of tracking scripts (web bugs) access user data each time a website is visited.

### User Tracking
- **Cookies**: Used for tracking user browsing behavior.
  - **First-Party Cookies**: Set by the website being visited.
  - **Third-Party Cookies**: Set by external services (e.g., advertisers) and used for tracking across different sites.
- **Trackers (Web Bugs)**: Small pieces of code that collect user activity data.

## 3. Progressive Loading of Web Pages

### Overview
- **Progressive Loading**: Technique to enhance user experience by rendering content progressively.

### HTTP
- Supports progressive rendering, allowing parts of a webpage to load without waiting for the entire file.

### JS/CSS
- JavaScript and CSS files often require the entire file to be processed, which can delay rendering if large.


# ACN - 14 Tutorial Session

Tutorial with multiple scenarios for Data transfer & retrieval from server.

# ACN - 14

## Privacy Concerns in Computer Networks

### 1. Tracking Browsing Sessions with Cookies
- **Cookies**: Small data files stored on the user's device that track browsing activities.
- **Purpose**: Enable personalized experiences, user authentication, and session management.

### 2. Disabling Trackers
- Users can take steps to disable trackers:
  - **Browser Settings**: Most browsers allow users to block or delete cookies.
  - **Privacy Extensions**: Tools like ad blockers and privacy-focused extensions can prevent trackers from functioning.
  - **Incognito/Private Mode**: Browsers offer private modes that limit cookie storage and tracking.

### 3. Understanding Referrer Data
- **Referrer**: The URL of the webpage that directed a user to the current page.
- **Importance**: 
  - Businesses analyze referrer data to understand traffic sources and optimize marketing strategies.
  - Knowing which sources yield the most engagement helps in refining advertising efforts.

### 4. Blocking Third-Party Cookies
- **Third-Party Cookies**: Set by domains other than the one the user is currently visiting, primarily used for tracking across websites.
- **Blocking Implications**:
  - When users block third-party cookies, they prevent tracking by advertisers and data aggregators.
  - **Important Note**: Blocking these cookies does not stop the data the website collects from the user. First-party data collection continues.

### 5. Broader Privacy Concerns
- **User Data Exploitation**: Collected data can be sold or shared without user consent, leading to privacy breaches.
- **Targeted Advertising**: While personalized ads can improve user experience, they raise concerns about excessive surveillance and data profiling.
- **Legal Frameworks**: Various laws (e.g., GDPR, CCPA) are being implemented to protect user privacy and regulate data collection practices.

### 6. Best Practices for Users
- **Regularly Clear Cookies**: Remove stored cookies periodically to minimize tracking.
- **Use Privacy-Focused Browsers**: Browsers like Brave or Firefox focus on enhancing user privacy.
- **Educate on Privacy Settings**: Understanding browser privacy settings and utilizing them can help protect personal information.

