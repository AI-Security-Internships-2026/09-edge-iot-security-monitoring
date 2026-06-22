## Edge-IIoTset

- **Source URL:** https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot
- **Licence:** Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
- **Version / date downloaded:** 2026-06-15
- **Size:** 1.6 GB compressed ZIP / 2,219,201 rows with 61 initial features (Targeted DNN Subset).
- **Format:** CSV + PCAP
- **Download command / script:** 
  "pip install -q kaggle
  kaggle datasets download -d mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot -f "Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"
- **Preprocessing steps:**
  1. Ingest the specialized deep learning dataset (DNN-EdgeIIoT-dataset.csv) comprising 1,638 mixed raw attributes.
  2. Strip environmental host identifiers and network markers to prevent overfitting: frame.time, ip.src_host, ip.dst_host, arp.src.proto_ipv4, arp.dst.proto_ipv4, http.file_data, http.request.full_uri, icmp.transmit_timestamp, http.request.uri.query, tcp.options, tcp.payload, tcp.srcport, tcp.dstport, udp.port, and mqtt.msg. 
  3. Execute rows sanitization using dropna(axis=0, how='any') and drop duplicated sequences using drop_duplicates(keep='first').
  4. Perform text dummy feature mapping on residual string parameters: http.request.method, http.referer, http.request.version, dns.qry.name.len, mqtt.conack.flags, mqtt.protoname, and mqtt.topic.
  5. Distribute structural matrix rows into non-overlapping client fragments to simulate distributed Non-IID topologies inside the Flower pipeline.
- **Train / Val / Test split:** 80% Training / 10% Validation / 10% Testing (Target framework distribution model)
- **Notes:** * Mandatory Academic Citation: Mohamed Amine Ferrag, Othmane Friha, Djallel Hamouda, Leandros Maglaras, Helge Janicke, "Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning", TechRxiv, 2022, Doi: 10.36227/techrxiv.18857336.v1  
The dataset files are stored strictly on local scratch space and are systematically ignored via `.gitignore` to comply with repository space restrictions.