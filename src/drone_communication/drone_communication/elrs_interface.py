import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float32
from sensor_msgs.msg import Joy  # Import Joy message type
import csv  # Import the csv module
import serial
import time
import argparse
from enum import IntEnum
import numpy as np
from datetime import datetime

from interfaces.msg import MotionCaptureState, ELRSCommand, Telemetry  # Import the Telemetry message

CRSF_SYNC = 0xC8

class PacketsTypes(IntEnum):
    GPS = 0x02
    VARIO = 0x07
    BATTERY_SENSOR = 0x08
    BARO_ALT = 0x09
    HEARTBEAT = 0x0B
    VIDEO_TRANSMITTER = 0x0F
    LINK_STATISTICS = 0x14
    RC_CHANNELS_PACKED = 0x16
    ATTITUDE = 0x1E
    FLIGHT_MODE = 0x21
    DEVICE_INFO = 0x29
    CONFIG_READ = 0x2C
    CONFIG_WRITE = 0x2D
    RADIO_ID = 0x3A

def crc8_dvb_s2(crc, a) -> int:
  crc = crc ^ a
  for ii in range(8):
    if crc & 0x80:
      crc = (crc << 1) ^ 0xD5
    else:
      crc = crc << 1
  return crc & 0xFF

def crc8_data(data) -> int:
    crc = 0
    for a in data:
        crc = crc8_dvb_s2(crc, a)
    return crc

def crsf_validate_frame(frame) -> bool:
    return crc8_data(frame[2:-1]) == frame[-1]

def signed_byte(b):
    return b - 256 if b >= 128 else b




def packCrsfToBytes(channels) -> bytes:
    # channels is in CRSF format! (0-1984)
    # Values are packed little-endianish such that bits BA987654321 -> 87654321, 00000BA9
    # 11 bits per channel x 16 channels = 22 bytes
    if len(channels) != 16:
        raise ValueError('CRSF must have 16 channels')
    result = bytearray()
    destShift = 0
    newVal = 0
    for ch in channels:
        # Put the low bits in any remaining dest capacity
        newVal |= (ch << destShift) & 0xff
        result.append(newVal)

        # Shift the high bits down and place them into the next dest byte
        srcBitsLeft = 11 - 8 + destShift
        newVal = ch >> (11 - srcBitsLeft)
        # When there's at least a full byte remaining, consume that as well
        if srcBitsLeft >= 8:
            result.append(newVal & 0xff)
            newVal >>= 8
            srcBitsLeft -= 8
        destShift = srcBitsLeft

    return result

def channelsCrsfToChannelsPacket(channels) -> bytes:
    result = bytearray([CRSF_SYNC, 24, PacketsTypes.RC_CHANNELS_PACKED]) # 24 is packet length
    result += packCrsfToBytes(channels)
    result.append(crc8_data(result[2:]))
    return result



import serial.tools.list_ports  # Import to list available serial ports

class ELRSInterface(Node):
    def __init__(self):
        super().__init__('elrs_interface')
        dt = 1 / 100
        self.timer = self.create_timer(dt, self.publish_message)
        self.telemetry_timer = self.create_timer(0.1, self.publish_telemetry)  # Timer for 10Hz publishing
        self.get_logger().info('ELRS Interface Node has started.')

        self.ser = None
        self.connect_serial()  # Initialize serial connection

        self.input = bytearray()

        self.idle = 993 # may be 993 for ardupilot (this is strange)
        self.range = 820
        self.armed = False
        self.packet = np.full(16, self.idle, dtype=np.uint16)
        self.packet[4] = 0

        self.battery_voltage = 0.0
        self.battery_mah_used = 0
        self.rssi = 0
        self.mode = "UNKNOWN"

        self.telemetry_publisher = self.create_publisher(Telemetry, 'telemetry', 10)  # Telemetry publisher

        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        self.subscription_floats = self.create_subscription(
            ELRSCommand, 'ELRSCommand', self.controller_commands_callback, qos_profile
        )

        self.last_message_time = time.time()  # Initialize the last message timestamp

        self.last_elrs_command_time_published = time.time()  # Initialize the last command timestamp

    def connect_serial(self):
        while self.ser is None:
            try:
                ports = serial.tools.list_ports.comports()
                for port in ports:
                    if '/dev/ttyUSB' in port.device:  # Check for USB serial devices
                        self.ser = serial.Serial(port.device, 921600, timeout=0.1)  # Reduced timeout
                        self.get_logger().info(f'Serial port {port.device} connected.')
                        return
                self.get_logger().warn('No suitable serial port found. Retrying...')
                time.sleep(1)  # Wait before retrying
            except serial.SerialException as e:
                self.get_logger().error(f'Error while connecting to serial port: {e}')
                time.sleep(1)  # Wait before retrying

    def controller_commands_callback(self, msg):
        self.last_message_time = time.time()  # Update the timestamp of the most recent message
        self.armed = bool(msg.armed)  # Ensure msg.armed is explicitly converted to a boolean

        self.get_logger().info(f"Armed: {self.armed}")

        self.packet[0] = self.idle + int(max(-1.0, min(1.0, msg.channel_0)) * self.range)
        self.packet[1] = self.idle + int(max(-1.0, min(1.0, msg.channel_1)) * self.range)
        self.packet[2] = self.idle + int(max(-1.0, min(1.0, msg.channel_2)) * self.range)
        #self.packet[2] = self.idle - self.range + int(max(0.0, min(1.0, msg.channel_2)) * 2 * self.range) #betaflight throttle
        print(f"Channel 2: {self.packet[2]} (msg.channel_2: {msg.channel_2})")

        self.packet[3] = self.idle + int(max(-1.0, min(1.0, msg.channel_3)) * self.range)
        self.packet[5] = self.idle + int(max(-1.0, min(1.0, msg.channel_4)) * self.range)
        self.packet[6] = self.idle + int(max(-1.0, min(1.0, msg.channel_5)) * self.range)
        self.packet[7] = self.idle + int(max(-1.0, min(1.0, msg.channel_6)) * self.range)
        self.packet[8] = self.idle + int(max(-1.0, min(1.0, msg.channel_7)) * self.range)
        self.packet[9] = self.idle + int(max(-1.0, min(1.0, msg.channel_8)) * self.range)
        self.packet[10] = self.idle + int(max(-1.0, min(1.0, msg.channel_9)) * self.range)
        self.packet[11] = self.idle + int(max(-1.0, min(1.0, msg.channel_10)) * self.range)

        if self.armed:
            self.packet[4] = 2000
        else:
            self.packet[4] = 0

    def handleCrsfPacket(self, ptype, data):
        if ptype == PacketsTypes.RADIO_ID and data[5] == 0x10:
            #print(f"OTX sync")
            pass
        elif ptype == PacketsTypes.LINK_STATISTICS:
            rssi1 = signed_byte(data[3])
            rssi2 = signed_byte(data[4])
            lq = data[5]
            snr = signed_byte(data[6])
            antenna = data[7]
            mode = data[8]
            power = data[9]
            # telemetry strength
            downlink_rssi = signed_byte(data[10])
            downlink_lq = data[11]
            downlink_snr = signed_byte(data[12])
            print(f"RSSI={rssi1}/{rssi2}dBm LQ={lq:03} mode={mode} ant={antenna} snr={snr} power={power} drssi={downlink_rssi} dlq={downlink_lq} dsnr={downlink_snr}")
            self.rssi = signed_byte(data[3])  # Update RSSI
        elif ptype == PacketsTypes.ATTITUDE:
            pitch = int.from_bytes(data[3:5], byteorder='big', signed=True) / 10000.0
            roll = int.from_bytes(data[5:7], byteorder='big', signed=True) / 10000.0
            yaw = int.from_bytes(data[7:9], byteorder='big', signed=True) / 10000.0
            print(f"Attitude: Pitch={pitch:0.2f} Roll={roll:0.2f} Yaw={yaw:0.2f} (rad)")
        elif ptype == PacketsTypes.FLIGHT_MODE:
            packet = ''.join(map(chr, data[3:-2]))
            print(f"Flight Mode: {packet}")
            self.mode = ''.join(map(chr, data[3:-2]))  # Update mode
        elif ptype == PacketsTypes.BATTERY_SENSOR:
            vbat = int.from_bytes(data[3:5], byteorder='big', signed=True) / 10.0
            curr = int.from_bytes(data[5:7], byteorder='big', signed=True) / 10.0
            mah = data[7] << 16 | data[8] << 7 | data[9]
            pct = data[10]
            print(f"Battery: {vbat:0.2f}V {curr:0.1f}A {mah}mAh {pct}%")
            self.battery_voltage = vbat  # Update battery voltage
            self.battery_mah_used = mah  # Update battery mAh used
        elif ptype == PacketsTypes.BARO_ALT:
            print(f"BaroAlt: ")
        elif ptype == PacketsTypes.DEVICE_INFO:
            packet = ' '.join(map(hex, data))
            print(f"Device Info: {packet}")
        elif data[2] == PacketsTypes.GPS:
            lat = int.from_bytes(data[3:7], byteorder='big', signed=True) / 1e7
            lon = int.from_bytes(data[7:11], byteorder='big', signed=True) / 1e7
            gspd = int.from_bytes(data[11:13], byteorder='big', signed=True) / 36.0
            hdg =  int.from_bytes(data[13:15], byteorder='big', signed=True) / 100.0
            alt = int.from_bytes(data[15:17], byteorder='big', signed=True) - 1000
            sats = data[17]
            print(f"GPS: Pos={lat} {lon} GSpd={gspd:0.1f}m/s Hdg={hdg:0.1f} Alt={alt}m Sats={sats}")
        elif ptype == PacketsTypes.VARIO:
            vspd = int.from_bytes(data[3:5], byteorder='big', signed=True) / 10.0
            print(f"VSpd: {vspd:0.1f}m/s")
        elif ptype == PacketsTypes.RC_CHANNELS_PACKED:
            pass
        else:
            packet = ' '.join(map(hex, data))
            print(f"Unknown 0x{ptype:02x}: {packet}")

    def publish_telemetry(self):
        # Publish telemetry data at 10Hz
        telemetry_msg = Telemetry()
        telemetry_msg.battery_voltage = self.battery_voltage
        telemetry_msg.battery_mah_used = self.battery_mah_used
        telemetry_msg.rssi = self.rssi
        telemetry_msg.mode = self.mode
        self.telemetry_publisher.publish(telemetry_msg)

    def publish_message(self):
        # Check if no message has been received for 0.1 seconds
        if time.time() - self.last_message_time > 0.5:
            if self.armed:  # Only log and disarm if currently armed
                self.get_logger().warn("No message received for 0.1 seconds. Disarming motors.")

            print(" -------------------------- THIS TRIGGERED --------------------------")
            self.armed = False
            self.packet = np.full(16, self.idle, dtype=np.uint16)
            self.packet[4] = 0

        try:
            if self.ser and self.ser.in_waiting > 0:
                self.input.extend(self.ser.read(self.ser.in_waiting))
            elif self.ser:
                self.ser.write(channelsCrsfToChannelsPacket(self.packet))

            while len(self.input) > 2:
                expected_len = self.input[1] + 2
                if expected_len > 64 or expected_len < 4:
                    self.input = bytearray()
                elif len(self.input) >= expected_len:
                    single = self.input[:expected_len]
                    self.input = self.input[expected_len:]

                    if not crsf_validate_frame(single):
                        packet = ' '.join(map(hex, single))
                        print(f"crc error: {packet}")
                    else:
                        self.handleCrsfPacket(single[2], single)
                else:
                    break
        except (serial.SerialException, OSError):
            self.get_logger().error('Serial connection lost. Attempting to reconnect...')
            self.ser = None
            self.connect_serial()
        


def main(args=None):
    rclpy.init(args=args)
    node = ELRSInterface()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
