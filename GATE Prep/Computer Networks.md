## Concepts of Layering: OSI & TCP/IP Protocols Stack

The concept of layering in networking refers to the division of the network architecture into discrete layers, each responsible for specific functions. This modular approach simplifies the design, implementation, and troubleshooting of network protocols. The two most widely recognized protocol stacks that use this concept are the OSI (Open Systems Interconnection) model and the TCP/IP (Transmission Control Protocol/Internet Protocol) model.

### OSI Model

The OSI model is a conceptual framework used to understand and design network systems. It consists of seven layers, each with specific functions and protocols. The layers are as follows (Down to Up Approach):

![OSI Model](../Archive/Attachment/OSI%20Model.jpg)

1. **Physical Layer (Layer 1):**
   - **Function:** Deals with the physical connection between devices and the transmission of binary data over physical media.
   - **Protocols/Technologies:** Ethernet, USB, Bluetooth, RS-232.

2. **Data Link Layer (Layer 2):**
   - **Function:** Provides node-to-node data transfer, error detection, and correction. It manages MAC (Media Access Control) addresses and ensures that data frames are error-free.
   - **Protocols/Technologies:** Ethernet, PPP (Point-to-Point Protocol), Switches, Bridges.

3. **Network Layer (Layer 3):**
   - **Function:** Manages data routing, forwarding, addressing, and packet switching. It determines the best path for data to travel from source to destination.
   - **Protocols/Technologies:** IP (Internet Protocol), ICMP (Internet Control Message Protocol), Routers.

4. **Transport Layer (Layer 4):**
   - **Function:** Provides reliable data transfer services to the upper layers. It manages flow control, error detection, and correction.
   - **Protocols/Technologies:** TCP (Transmission Control Protocol), UDP (User Datagram Protocol).

5. **Session Layer (Layer 5):**
   - **Function:** Manages sessions and connections between applications. It establishes, maintains, and terminates sessions.
   - **Protocols/Technologies:** NetBIOS, RPC (Remote Procedure Call).

6. **Presentation Layer (Layer 6):**
   - **Function:** Translates data between the application layer and the network format. It manages data encryption, decryption, compression, and translation.
   - **Protocols/Technologies:** SSL/TLS (Secure Sockets Layer/Transport Layer Security), JPEG, MPEG.

7. **Application Layer (Layer 7):**
   - **Function:** Provides network services directly to end-user applications. It facilitates application-specific network operations.
   - **Protocols/Technologies:** HTTP, FTP, SMTP, DNS.

### TCP/IP Model

The Current TCP/IP model is depicted with 5 layers to include the Network Access layer as two separate layers: Physical and Data Link. Here’s the 5-layer version of the TCP/IP model:

![TCP/IP Model](../Archive/Attachment/TCPIP%20Model-1.png)

1. **Physical Layer:**
   - **Function:** Deals with the physical connection between devices and the transmission of raw bit streams over a physical medium.
   - **Examples:** Ethernet cables, Wi-Fi, modems, network interface cards (NICs).

2. **Data Link Layer:**
   - **Function:** Provides node-to-node data transfer, error detection, and correction. It handles MAC addresses and framing.
   - **Examples:** Ethernet, PPP (Point-to-Point Protocol), switches.

3. **Network Layer:**
   - **Function:** Manages logical addressing, routing, and packet forwarding. It determines the best path for data to travel.
   - **Examples:** IP (Internet Protocol), ICMP (Internet Control Message Protocol), routers.

4. **Transport Layer:**
   - **Function:** Provides reliable data transfer services to the upper layers, including flow control, error detection, and correction.
   - **Examples:** TCP (Transmission Control Protocol), UDP (User Datagram Protocol).

5. **Application Layer:**
   - **Function:** Provides network services directly to end-user applications. It includes high-level protocols used by applications for communication.
   - **Examples:** HTTP, FTP, SMTP, DNS, Telnet.

### Comparison and Key Points

- **Layering Concept:** Both models use a layered approach to separate concerns and functions, making network design and troubleshooting more manageable.
- **Development Purpose:** The OSI model is more of a theoretical framework, while the TCP/IP model was developed based on practical implementation.
- **Layer Functions:** Each layer in both models serves distinct functions, though the OSI model provides a more detailed separation of responsibilities.

## Routers & Switches

### **Routers**:
- **Function:** Direct data between different networks.
- **Use:** Connect multiple networks, determine the best path for data to travel.
- **Example:** Connecting a home network to the internet.

### **Switches**:
- **Function:** Connect devices within the same network.
- **Use:** Forward data to specific devices based on MAC addresses.
- **Example:** Connecting computers within an office network.

### **Difference**:
- **Scope:** Routers operate at the network layer (Layer 3) and can route data between different networks. Switches operate at the data link layer (Layer 2) and manage data flow within a single network.
- **Addressing:** Routers use IP addresses to route data. Switches use MAC addresses to forward data.
- Routers Connect to multiple switches and then switches are used for Internal Network. (IIT Hyderabad uses Switches inside Lecture Hall)
- **Scope:** Routers provide broader security controls for entire networks, including external traffic. Switches focus on internal network security.
- **Threat Protection:** Routers offer more comprehensive protection against external threats, while switches prevent unauthorized access within the network.
- **Tools:** Routers use firewalls, ACLs, and NAT, whereas switches use port security, VLANs, and MAC filtering.