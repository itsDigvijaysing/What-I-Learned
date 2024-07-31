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

The TCP/IP model is a more simplified and practical framework compared to the OSI model. It has four layers, each corresponding to one or more OSI layers:

![TCP/IP Model](../Archive/Attachment/TCPIP%20Model-1.png)

1. **Network Interface Layer (Link Layer):**
   - **Function:** Corresponds to the OSI's Physical and Data Link layers. It handles the physical transmission of data over network media.
   - **Protocols/Technologies:** Ethernet, Wi-Fi, ARP (Address Resolution Protocol).

2. **Internet Layer:**
   - **Function:** Corresponds to the OSI's Network layer. It manages logical addressing, routing, and packet forwarding.
   - **Protocols/Technologies:** IP, ICMP, IGMP (Internet Group Management Protocol).

3. **Transport Layer:**
   - **Function:** Corresponds to the OSI's Transport layer. It provides end-to-end communication services, error handling, and flow control.
   - **Protocols/Technologies:** TCP, UDP.

4. **Application Layer:**
   - **Function:** Corresponds to the OSI's Session, Presentation, and Application layers. It includes high-level protocols used by applications for network communication.
   - **Protocols/Technologies:** HTTP, FTP, SMTP, DNS, Telnet.

### Comparison and Key Points

- **Layering Concept:** Both models use a layered approach to separate concerns and functions, making network design and troubleshooting more manageable.
- **Number of Layers:** The OSI model has seven layers, while the TCP/IP model has four.
- **Development Purpose:** The OSI model is more of a theoretical framework, while the TCP/IP model was developed based on practical implementation.
- **Layer Functions:** Each layer in both models serves distinct functions, though the OSI model provides a more detailed separation of responsibilities.
