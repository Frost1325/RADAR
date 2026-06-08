databases is typically

21 days prior to the effective date of the revision.

The integrity of the data is ensured through a process called cyclic redundancy check (CRC). A CRC is an error detection algorithm capable of detecting small bit- level changes in a block of data. The CRC algorithm treats a data block as a single, large binary value. The data block is divided by a fixed binary number called a generator polynomial whose form and magnitude is determined based on the level of integrity desired. The remainder of the division is the CRC value for the data block. This value is stored and transmitted with the corresponding data block. The integrity of the data is checked by reapplying the CRC algorithm prior to distribution.