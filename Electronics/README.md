# Communications protocols

- [Serial communication](https://en.wikipedia.org/wiki/Serial_communication)
- [Parallel communication](https://en.wikipedia.org/wiki/Parallel_communication)

# Intra-board Communications

- [I2C Bus](https://en.wikipedia.org/wiki/I%C2%B2C)
  - Serial: one bit at a time
  - Synchronous: clk signal
  - Multi-master, Multi-slave
  - Packet switched: header (which addresses the slave) + payload (the data)
  - Two wire single-ended: one wire GND the other contains the signal
  - Half duplex: allows simultaneous and bidirectional communication
- [SPI Bus](https://en.wikipedia.org/wiki/Serial_Peripheral_Interface)
  - Serial: one bit at a time
  - Synchronous: clk signal
  - Single-master, Multi-slave
  - Four wire differential signaling: use two complementary signals, permits cancelling noise
  - Full duplex: allows simultaneous and bidirectional communication
  