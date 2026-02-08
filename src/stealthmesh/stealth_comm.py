"""
Stealth Communication Module for StealthMesh
Implements polymorphic encryption, packet camouflage, and covert channels
"""

import os
import json
import base64
import hashlib
import secrets
import struct
import time
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend


class CipherType(Enum):
    """Available cipher types for polymorphic encryption"""
    AES_256_GCM = "aes_256_gcm"
    AES_256_CBC = "aes_256_cbc"
    CHACHA20 = "chacha20"


class CamouflageType(Enum):
    """Packet camouflage techniques"""
    HTTP_MIMICRY = "http"
    DNS_TUNNELING = "dns"
    HTTPS_MIMICRY = "https"
    ICMP_COVERT = "icmp"


@dataclass
class StealthPacket:
    """Encrypted and camouflaged packet structure"""
    payload: bytes
    cipher_type: CipherType
    camouflage_type: CamouflageType
    nonce: bytes
    tag: Optional[bytes]
    timestamp: float
    sequence: int
    checksum: str


class PolymorphicEncryption:
    """
    Polymorphic encryption that rotates ciphers and keys
    to evade pattern-based detection
    """
    
    def __init__(self, master_key: bytes = None, rotation_interval: int = 100):
        """
        Initialize polymorphic encryption
        
        Args:
            master_key: Master secret key (32 bytes)
            rotation_interval: Number of encryptions before key rotation
        """
        self.master_key = master_key or secrets.token_bytes(32)
        self.rotation_interval = rotation_interval
        self.encryption_count = 0
        self.current_key = self._derive_key(self.master_key, b"initial")
        self.cipher_sequence = list(CipherType)
        self.current_cipher_idx = 0
        
    def _derive_key(self, master: bytes, salt: bytes) -> bytes:
        """Derive a key from master key using PBKDF2"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        return kdf.derive(master)
    
    def _rotate_key(self):
        """Rotate encryption key"""
        salt = secrets.token_bytes(16)
        self.current_key = self._derive_key(self.master_key, salt)
        self.current_cipher_idx = (self.current_cipher_idx + 1) % len(self.cipher_sequence)
        
    def get_current_cipher(self) -> CipherType:
        """Get current cipher in rotation"""
        return self.cipher_sequence[self.current_cipher_idx]
    
    def encrypt(self, plaintext: bytes) -> Tuple[bytes, bytes, Optional[bytes], CipherType]:
        """
        Encrypt data with current cipher
        
        Args:
            plaintext: Data to encrypt
            
        Returns:
            Tuple of (ciphertext, nonce, tag, cipher_type)
        """
        # Check if rotation needed
        self.encryption_count += 1
        if self.encryption_count >= self.rotation_interval:
            self._rotate_key()
            self.encryption_count = 0
            
        cipher_type = self.get_current_cipher()
        
        if cipher_type == CipherType.AES_256_GCM:
            return self._encrypt_aes_gcm(plaintext)
        elif cipher_type == CipherType.AES_256_CBC:
            return self._encrypt_aes_cbc(plaintext)
        elif cipher_type == CipherType.CHACHA20:
            return self._encrypt_chacha20(plaintext)
            
    def _encrypt_aes_gcm(self, plaintext: bytes) -> Tuple[bytes, bytes, bytes, CipherType]:
        """AES-256-GCM encryption"""
        nonce = secrets.token_bytes(12)
        cipher = Cipher(
            algorithms.AES(self.current_key),
            modes.GCM(nonce),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        return ciphertext, nonce, encryptor.tag, CipherType.AES_256_GCM
    
    def _encrypt_aes_cbc(self, plaintext: bytes) -> Tuple[bytes, bytes, None, CipherType]:
        """AES-256-CBC encryption with PKCS7 padding"""
        iv = secrets.token_bytes(16)
        
        # Pad plaintext
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(plaintext) + padder.finalize()
        
        cipher = Cipher(
            algorithms.AES(self.current_key),
            modes.CBC(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        return ciphertext, iv, None, CipherType.AES_256_CBC
    
    def _encrypt_chacha20(self, plaintext: bytes) -> Tuple[bytes, bytes, None, CipherType]:
        """ChaCha20 encryption"""
        nonce = secrets.token_bytes(16)
        cipher = Cipher(
            algorithms.ChaCha20(self.current_key, nonce),
            mode=None,
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        return ciphertext, nonce, None, CipherType.CHACHA20
    
    def decrypt(self, ciphertext: bytes, nonce: bytes, tag: Optional[bytes], 
                cipher_type: CipherType) -> bytes:
        """
        Decrypt data
        
        Args:
            ciphertext: Encrypted data
            nonce: Nonce/IV used for encryption
            tag: Authentication tag (for GCM)
            cipher_type: Cipher used for encryption
            
        Returns:
            Decrypted plaintext
        """
        if cipher_type == CipherType.AES_256_GCM:
            return self._decrypt_aes_gcm(ciphertext, nonce, tag)
        elif cipher_type == CipherType.AES_256_CBC:
            return self._decrypt_aes_cbc(ciphertext, nonce)
        elif cipher_type == CipherType.CHACHA20:
            return self._decrypt_chacha20(ciphertext, nonce)
            
    def _decrypt_aes_gcm(self, ciphertext: bytes, nonce: bytes, tag: bytes) -> bytes:
        """AES-256-GCM decryption"""
        cipher = Cipher(
            algorithms.AES(self.current_key),
            modes.GCM(nonce, tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()
    
    def _decrypt_aes_cbc(self, ciphertext: bytes, iv: bytes) -> bytes:
        """AES-256-CBC decryption"""
        cipher = Cipher(
            algorithms.AES(self.current_key),
            modes.CBC(iv),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Remove padding
        unpadder = padding.PKCS7(128).unpadder()
        return unpadder.update(padded_data) + unpadder.finalize()
    
    def _decrypt_chacha20(self, ciphertext: bytes, nonce: bytes) -> bytes:
        """ChaCha20 decryption"""
        cipher = Cipher(
            algorithms.ChaCha20(self.current_key, nonce),
            mode=None,
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()


class PacketCamouflage:
    """
    Camouflage encrypted traffic to look like legitimate protocols
    """
    
    HTTP_HEADERS = [
        "GET / HTTP/1.1",
        "POST /api/data HTTP/1.1",
        "GET /images/logo.png HTTP/1.1",
        "POST /login HTTP/1.1",
    ]
    
    DNS_DOMAINS = [
        "cdn.example.com",
        "api.services.net",
        "static.content.org",
        "update.software.io",
    ]
    
    def __init__(self, camouflage_type: CamouflageType = CamouflageType.HTTP_MIMICRY):
        self.camouflage_type = camouflage_type
        
    def camouflage(self, data: bytes) -> bytes:
        """
        Wrap encrypted data in protocol-mimicking format
        
        Args:
            data: Encrypted payload
            
        Returns:
            Camouflaged packet
        """
        if self.camouflage_type == CamouflageType.HTTP_MIMICRY:
            return self._http_camouflage(data)
        elif self.camouflage_type == CamouflageType.DNS_TUNNELING:
            return self._dns_camouflage(data)
        elif self.camouflage_type == CamouflageType.HTTPS_MIMICRY:
            return self._https_camouflage(data)
        elif self.camouflage_type == CamouflageType.ICMP_COVERT:
            return self._icmp_camouflage(data)
            
    def _http_camouflage(self, data: bytes) -> bytes:
        """Disguise as HTTP traffic"""
        header = secrets.choice(self.HTTP_HEADERS)
        encoded_data = base64.b64encode(data).decode()
        
        http_packet = f"""{header}
Host: {secrets.choice(self.DNS_DOMAINS)}
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)
Content-Type: application/octet-stream
Content-Length: {len(encoded_data)}
X-Request-ID: {secrets.token_hex(16)}

{encoded_data}"""
        
        return http_packet.encode()
    
    def _dns_camouflage(self, data: bytes) -> bytes:
        """Disguise as DNS query (data in subdomain labels)"""
        # Encode data as hex and split into labels
        hex_data = data.hex()
        labels = [hex_data[i:i+63] for i in range(0, len(hex_data), 63)]
        
        # Create DNS-like structure
        domain = secrets.choice(self.DNS_DOMAINS)
        subdomain = ".".join(labels[:4])  # Max 4 labels
        
        dns_packet = {
            "query": f"{subdomain}.{domain}",
            "type": "TXT",
            "id": secrets.randbelow(65535),
            "flags": 0x0100,
            "data_overflow": labels[4:] if len(labels) > 4 else []
        }
        
        return json.dumps(dns_packet).encode()
    
    def _https_camouflage(self, data: bytes) -> bytes:
        """Disguise as HTTPS-like TLS record"""
        # Mimic TLS Application Data record
        tls_header = bytes([
            0x17,  # Application Data
            0x03, 0x03,  # TLS 1.2
        ])
        length = struct.pack(">H", len(data))
        
        return tls_header + length + data
    
    def _icmp_camouflage(self, data: bytes) -> bytes:
        """Disguise as ICMP echo (ping) packet"""
        icmp_header = bytes([
            0x08,  # Echo request
            0x00,  # Code
            0x00, 0x00,  # Checksum placeholder
            0x00, 0x01,  # Identifier
            0x00, 0x01,  # Sequence
        ])
        
        return icmp_header + data
    
    def decamouflage(self, packet: bytes) -> bytes:
        """
        Extract encrypted data from camouflaged packet
        
        Args:
            packet: Camouflaged packet
            
        Returns:
            Encrypted payload
        """
        if self.camouflage_type == CamouflageType.HTTP_MIMICRY:
            return self._http_decamouflage(packet)
        elif self.camouflage_type == CamouflageType.DNS_TUNNELING:
            return self._dns_decamouflage(packet)
        elif self.camouflage_type == CamouflageType.HTTPS_MIMICRY:
            return self._https_decamouflage(packet)
        elif self.camouflage_type == CamouflageType.ICMP_COVERT:
            return self._icmp_decamouflage(packet)
            
    def _http_decamouflage(self, packet: bytes) -> bytes:
        """Extract data from HTTP-like packet"""
        text = packet.decode()
        # Find the body after double newline
        parts = text.split("\n\n", 1)
        if len(parts) == 2:
            return base64.b64decode(parts[1])
        return b""
    
    def _dns_decamouflage(self, packet: bytes) -> bytes:
        """Extract data from DNS-like packet"""
        dns_data = json.loads(packet.decode())
        query = dns_data["query"]
        
        # Extract subdomain labels
        parts = query.split(".")
        domain_parts = 2  # example.com
        labels = parts[:-domain_parts]
        
        # Add overflow data
        labels.extend(dns_data.get("data_overflow", []))
        
        hex_data = "".join(labels)
        return bytes.fromhex(hex_data)
    
    def _https_decamouflage(self, packet: bytes) -> bytes:
        """Extract data from TLS-like record"""
        # Skip TLS header (5 bytes)
        return packet[5:]
    
    def _icmp_decamouflage(self, packet: bytes) -> bytes:
        """Extract data from ICMP-like packet"""
        # Skip ICMP header (8 bytes)
        return packet[8:]


class StealthCommunication:
    """
    Main stealth communication interface combining
    polymorphic encryption and packet camouflage
    """
    
    def __init__(self, 
                 master_key: bytes = None,
                 camouflage_type: CamouflageType = CamouflageType.HTTP_MIMICRY,
                 rotation_interval: int = 100):
        """
        Initialize stealth communication
        
        Args:
            master_key: 32-byte master key
            camouflage_type: Type of traffic camouflage
            rotation_interval: Key rotation interval
        """
        self.encryption = PolymorphicEncryption(master_key, rotation_interval)
        self.camouflage = PacketCamouflage(camouflage_type)
        self.sequence = 0
        
    def create_stealth_packet(self, data: Dict[str, Any]) -> StealthPacket:
        """
        Create a fully encrypted and camouflaged packet
        
        Args:
            data: Dictionary of data to send
            
        Returns:
            StealthPacket ready for transmission
        """
        # Serialize data
        plaintext = json.dumps(data).encode()
        
        # Encrypt
        ciphertext, nonce, tag, cipher_type = self.encryption.encrypt(plaintext)
        
        # Camouflage
        camouflaged = self.camouflage.camouflage(ciphertext)
        
        # Create packet
        self.sequence += 1
        packet = StealthPacket(
            payload=camouflaged,
            cipher_type=cipher_type,
            camouflage_type=self.camouflage.camouflage_type,
            nonce=nonce,
            tag=tag,
            timestamp=time.time(),
            sequence=self.sequence,
            checksum=hashlib.sha256(camouflaged).hexdigest()[:16]
        )
        
        return packet
    
    def decode_stealth_packet(self, packet: StealthPacket) -> Dict[str, Any]:
        """
        Decode a received stealth packet
        
        Args:
            packet: Received StealthPacket
            
        Returns:
            Original data dictionary
        """
        # Verify checksum
        calculated_checksum = hashlib.sha256(packet.payload).hexdigest()[:16]
        if calculated_checksum != packet.checksum:
            raise ValueError("Packet checksum mismatch - possible tampering")
            
        # Decamouflage
        ciphertext = self.camouflage.decamouflage(packet.payload)
        
        # Decrypt
        plaintext = self.encryption.decrypt(
            ciphertext, 
            packet.nonce, 
            packet.tag,
            packet.cipher_type
        )
        
        return json.loads(plaintext.decode())
    
    def serialize_packet(self, packet: StealthPacket) -> bytes:
        """Serialize packet for network transmission"""
        packet_dict = {
            "payload": base64.b64encode(packet.payload).decode(),
            "cipher": packet.cipher_type.value,
            "camouflage": packet.camouflage_type.value,
            "nonce": base64.b64encode(packet.nonce).decode(),
            "tag": base64.b64encode(packet.tag).decode() if packet.tag else None,
            "timestamp": packet.timestamp,
            "sequence": packet.sequence,
            "checksum": packet.checksum
        }
        return json.dumps(packet_dict).encode()
    
    def deserialize_packet(self, data: bytes) -> StealthPacket:
        """Deserialize packet from network"""
        packet_dict = json.loads(data.decode())
        return StealthPacket(
            payload=base64.b64decode(packet_dict["payload"]),
            cipher_type=CipherType(packet_dict["cipher"]),
            camouflage_type=CamouflageType(packet_dict["camouflage"]),
            nonce=base64.b64decode(packet_dict["nonce"]),
            tag=base64.b64decode(packet_dict["tag"]) if packet_dict["tag"] else None,
            timestamp=packet_dict["timestamp"],
            sequence=packet_dict["sequence"],
            checksum=packet_dict["checksum"]
        )


# Utility functions
def generate_node_key() -> bytes:
    """Generate a secure 32-byte key for a mesh node"""
    return secrets.token_bytes(32)


def derive_shared_key(node_key: bytes, peer_id: str) -> bytes:
    """Derive a shared key for peer-to-peer communication"""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=peer_id.encode(),
        iterations=100000,
        backend=default_backend()
    )
    return kdf.derive(node_key)


if __name__ == "__main__":
    # Test stealth communication
    print("Testing StealthMesh Stealth Communication Module")
    print("=" * 60)
    
    # Create communicator
    key = generate_node_key()
    comm = StealthCommunication(
        master_key=key,
        camouflage_type=CamouflageType.HTTP_MIMICRY,
        rotation_interval=5
    )
    
    # Test message
    test_data = {
        "alert_type": "intrusion_detected",
        "source_ip": "192.168.1.100",
        "threat_level": "high",
        "timestamp": time.time()
    }
    
    print(f"\nOriginal data: {test_data}")
    
    # Create stealth packet
    packet = comm.create_stealth_packet(test_data)
    print(f"\nPacket created:")
    print(f"  - Cipher: {packet.cipher_type.value}")
    print(f"  - Camouflage: {packet.camouflage_type.value}")
    print(f"  - Payload size: {len(packet.payload)} bytes")
    print(f"  - Sequence: {packet.sequence}")
    
    # Serialize for transmission
    serialized = comm.serialize_packet(packet)
    print(f"\nSerialized size: {len(serialized)} bytes")
    
    # Deserialize and decode
    received_packet = comm.deserialize_packet(serialized)
    decoded_data = comm.decode_stealth_packet(received_packet)
    print(f"\nDecoded data: {decoded_data}")
    
    # Verify
    assert test_data == decoded_data, "Data mismatch!"
    print("\n✓ Stealth communication test passed!")
    
    # Test cipher rotation
    print("\n" + "=" * 60)
    print("Testing cipher rotation...")
    for i in range(7):
        packet = comm.create_stealth_packet({"msg": f"test_{i}"})
        print(f"  Message {i}: {packet.cipher_type.value}")
