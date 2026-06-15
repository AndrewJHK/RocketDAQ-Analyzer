import csv
import os
from datetime import datetime
from pathlib import Path
from pymongo import MongoClient, ASCENDING
from bson import decode_file_iter
from bson.raw_bson import RawBSONDocument
from collections import namedtuple
from src.processing_utils import logger

DeviceCSVConfig = namedtuple("DeviceCSVConfig", ["name", "origin_id", "fieldnames", "channel_mapping"])

DEVICE_NAME_MAPPING = {
    100: "lpb",
    130: "adv_usb",
    131: "adv_pcie",
    200: "comp"
}
CHANNEL_LABEL_MAPPINGS = {
    130: {
        "usb4716.chan0.scaled": "N2O",
        "usb4716.chan1.scaled": "CHAMBER_PRES",
        "usb4716.chan2.scaled": "N2O_PRES",
        "usb4716.chan3.scaled": "FUEL"
    },
    131: {
        "pcie1816.chan1.scaled": "GSE_N2O_IN",
        "pcie1816.chan2.scaled": "GSE_N2",
        "pcie1816.chan3.scaled": "COPV",
        "pcie1816.chan4.scaled": "Oxidizer_TANK",
        "pcie1816.chan6.scaled": "GSE_N2O_OUT",
        "pcie1816.chan7.scaled": "FUEL_TANK",
        "pcie1816.chan9.scaled": "REF",
    },
    100: {
        "adc1.chan0.scaled": "TM2.scaled",
        "adc1.chan0.raw": "TM2.raw",
        "adc1.chan1.scaled": "PT2.scaled",
        "adc1.chan1.raw": "PT2.raw",
        "adc1.chan2.scaled": "PT1.scaled",
        "adc1.chan2.raw": "PT1.raw",
        "adc1.chan3.scaled": "TM1.scaled",
        "adc1.chan3.raw": "TM1.raw",
        "adc2.chan0.scaled": "PT5.scaled",
        "adc2.chan0.raw": "PT5.raw",
        "adc2.chan1.scaled": "PT6.scaled",
        "adc2.chan1.raw": "PT6.raw",
        "adc2.chan2.scaled": "PT4.scaled",
        "adc2.chan2.raw": "PT4.raw",
        "adc2.chan3.scaled": "PT3.scaled",
        "adc2.chan3.raw": "PT3.raw"

    }
}

_WRITE_BATCH = 20000


class MongoDBDataRetriever:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.databases = {}
        try:
            self.client = MongoClient(f"mongodb://{self.ip}:{self.port}/", document_class=RawBSONDocument)
            logger.info("Succesfully setup a MongoClient")
        except Exception as e:
            logger.error(f"MongoClient setup failed due to an error:{e}")

    def change_client(self, ip: str, port: int):
        """
        Change the ip and port for the current client

        :param ip:
        :param port:
        :return:
        """
        self.client.close()
        self.client = MongoClient(f"mongodb://{ip}:{port}/", document_class=RawBSONDocument)

    def retrieve_databases(self):
        """
        Update the list of databases and collections
        """
        try:
            self.databases = {
                db_name: self.client[db_name].list_collection_names()
                for db_name in self.client.list_database_names()
            }
        except Exception as e:
            logger.error(f"Failed to retrieve databases: {e}")
            self.databases = {}

    def get_databases(self) -> dict:
        """
        Get all the loaded databases and their collections

        :return: dict
        """
        return self.databases

    def get_collections_in_database(self, db: str) -> list:
        """
        Get all collections from the specified database

        :param db:
        :return: list
        """
        return self.databases.get(db, [])

    def get_doc_amount(self, db: str, collection: str) -> int:
        """
        Get the amount of docs in the selected collection

        :param db:
        :param collection:
        :return: int
        """
        try:
            return self.client[db][collection].count_documents({})
        except Exception as e:
            logger.error(f"Failed to count documents in {db}.{collection}: {e}")
            return 0

    def retrieve_bson_range(self, db: str, collection: str, filename: str, start: int = 0, stop: int = None):

        """
        Get all docs from the databases collection in said range and save to a .bson file in the same format as mongodump

        :param db:           Database name
        :param collection:   Collection name (time-series)
        :param filename:     Output .bson file path
        :param start:        Start number of doc (inclusive).
        :param stop:         Stop number of doc (exclusive).
        """

        cursor = self.client[db][collection].find().skip(start)
        if stop is not None:
            cursor = cursor.limit(max(0, stop - start))

        try:
            dest = Path(filename)

            if dest.parent == Path('.'):
                dest = Path('data') / dest.name

            dest.parent.mkdir(parents=True, exist_ok=True)

            with dest.open("wb") as f:
                for doc in cursor:
                    f.write(doc.raw)

            logger.info(f"Successfully dumped collection: {collection} to {dest}")
            return str(dest)
        except Exception as e:
            logger.error(f"Failed to dump collection: {collection} to {filename}: {e}")
            return None

    def retrieve_bson_time_range(self, db: str, collection: str, filename: str, start=None, stop=None):
        """
        Get all docs from the databases collection in said timeseries range and save to a .bson file in the same format as mongodump

        :param db:           Database name
        :param collection:   Collection name (time-series)
        :param filename:     Output .bson file path
        :param start:        Start time (inclusive).
        :param stop:         Stop time (exclusive).
        """

        query = {}
        if start is not None or stop is not None:
            time_cond = {}
            if start is not None:
                time_cond["$gte"] = start
            if stop is not None:
                time_cond["$lt"] = stop
            query["ts"] = time_cond

        cursor = (self.client[db][collection].find(query).sort("ts", ASCENDING))

        try:
            dest = Path(filename)

            if dest.parent == Path('.'):
                dest = Path('data') / dest.name

            dest.parent.mkdir(parents=True, exist_ok=True)

            with dest.open("wb") as f:
                for doc in cursor:
                    f.write(doc.raw)

            logger.info(
                f"Successfully dumped time-range from {db}.{collection} to {dest}"
            )
            return str(dest)
        except Exception as e:
            logger.error(
                f"Failed to dump time-range from {db}.{collection} to {filename}: {e}"
            )
            return None


class DATAParser:
    """
    Parser for JSON/BSON data into CSV with incremental processing to minimize memory usage.
    """

    def __init__(self, csv_path=None, interpolated=True, bson_file=None):
        """
        Initialize the parser from a BSON file.

        :param csv_path: path where CSV files will be written
        :param interpolated: whether to fill missing values (True) or leave them empty (False)
        :param bson_file: path to BSON file for incremental processing
        """
        self.csv_path = csv_path
        self.interpolated = interpolated
        self.bson_file = bson_file
        self.fields_per_origin = {}
        self.last_known = {}
        self.counters = {}
        self.devices = {}
        self._non_header_keys = {}

        if bson_file is not None:
            self._scan_bson_for_metadata()
            self._initialize_devices()

    def _scan_bson_for_metadata(self, chunk_size=10000):
        """
        Scan BSON file in chunks to extract field metadata without loading entire file.
        Only reads first chunk_size records to discover all fields efficiently.

        :param chunk_size: number of records to scan for metadata discovery
        """
        try:
            with open(self.bson_file, "rb") as f:
                count = 0
                for doc in decode_file_iter(f):
                    origin = doc.get("data", {}).get("header", {}).get("origin")
                    if origin is None:
                        continue
                    if origin not in self.fields_per_origin:
                        self.fields_per_origin[origin] = set()
                    flat = self.flatten_dict(doc.get("data", {}).get("data", {}))
                    for full_key in flat:
                        mapped_key = self.map_key(origin, full_key)
                        self.fields_per_origin[origin].add(f"data.{mapped_key}")

                    count += 1
                    if count >= chunk_size:
                        logger.debug(f"Metadata scan completed on {count} records")
                        break
        except Exception as e:
            logger.error(f"Failed to scan BSON metadata: {e}")
            raise

    def _initialize_devices(self):
        """
        Initialize CSV configurations and counters for each origin.
        """
        for origin, fields in self.fields_per_origin.items():
            field_list = ["header.origin", "header.timestamp_epoch", "header.timestamp_human",
                          "header.counter"] + sorted(list(fields))
            dev_name = DEVICE_NAME_MAPPING.get(origin, f"dev_{origin}")
            self.devices[origin] = DeviceCSVConfig(dev_name, origin, field_list, CHANNEL_LABEL_MAPPINGS.get(origin, {}))
            self.last_known[origin] = {}
            self.counters[origin] = 0
            self._non_header_keys[origin] = [
                f for f in field_list
                if not f.startswith("header") and f != "data.cpu_temperature"
            ]

    def _build_row(self, record, origin):
        """
        Build a CSV row dict from a BSON record.
        Returns the row dict, or None if the row should be skipped.
        """
        rec_data = record["data"]
        ts_ms = int(float(rec_data["header"].get("timestamp", "1000190760000")))
        try:
            base_ts = datetime.fromtimestamp(ts_ms // 1000)
            ts_human = f"{base_ts.strftime('%Y-%m-%d %H:%M:%S')}.{ts_ms}"
        except (OSError, ValueError):
            ts_human = "2001-09-11 08:46:00.000"

        # Start from last_known — gives interpolation for free, no fill-missing loop needed
        row = dict(self.last_known[origin])
        row["header.origin"] = origin
        row["header.timestamp_epoch"] = ts_ms
        row["header.timestamp_human"] = ts_human
        row["header.counter"] = self.counters[origin]

        mapping = CHANNEL_LABEL_MAPPINGS.get(origin, {})
        last = self.last_known[origin]
        for field_key, value in rec_data.get("data", {}).items():
            full_key = f"data.{mapping.get(field_key, field_key)}"
            last[full_key] = value
            row[full_key] = value

        if not self.interpolated:
            for k in self._non_header_keys[origin]:
                if k not in rec_data.get("data", {}):
                    row[k] = None

        if all(row.get(k) is None for k in self._non_header_keys[origin]):
            return None

        self.counters[origin] += 1
        return row

    def json_to_csv(self):
        """
        Export all JSON data into CSV files (one per origin).
        Supports both legacy mode (pre-loaded data with sorting) and incremental BSON mode (streamed processing).

        :return: list of paths to generated CSV files
        """
        suffix = "_interpolated" if self.interpolated else "_none_filled"
        file_paths = {
            origin: f"{self.csv_path}{suffix}_{device.name}.csv"
            for origin, device in self.devices.items()
        }

        writers = {}
        files = {}
        try:
            # Initialize CSV files and writers
            for origin, device in self.devices.items():
                file = open(file_paths[origin], mode='w', newline='', buffering=1 << 23)
                writer = csv.DictWriter(file, fieldnames=device.fieldnames)
                writer.writeheader()
                self.last_known[origin] = {field: None for field in device.fieldnames}
                writers[origin] = writer
                files[origin] = file

            # Incremental BSON processing - stream through file without loading all records
            self._write_records_from_bson(writers, file_paths)

        finally:
            for file in files.values():
                file.close()

        # Clean up empty files and remove empty columns
        self._cleanup_csv_files(file_paths)

        return list(file_paths.values())

    def _write_records_from_bson(self, writers, file_paths):
        """
        Stream records from BSON file and write to CSV in batches.

        :param writers: dict of csv.DictWriter objects
        :param file_paths: dict of file paths for logging
        """
        batches = {origin: [] for origin in self.devices}
        try:
            with open(self.bson_file, "rb") as f:
                for doc in decode_file_iter(f):
                    origin = doc.get("data", {}).get("header", {}).get("origin")
                    if origin not in self.devices:
                        continue
                    row = self._build_row(doc, origin)
                    if row is None:
                        continue
                    batch = batches[origin]
                    batch.append(row)
                    if len(batch) >= self._WRITE_BATCH:
                        writers[origin].writerows(batch)
                        batch.clear()

            for origin, batch in batches.items():
                if batch:
                    writers[origin].writerows(batch)

        except Exception as e:
            logger.error(f"Error writing records from BSON: {e}")
            raise

    def _cleanup_csv_files(self, file_paths):
        """
        Remove empty CSV files and drop empty columns from remaining files.
        Uses two streaming passes to avoid loading the entire file into memory.

        :param file_paths: dict of origin -> file path
        """
        for key, path in list(file_paths.items()):
            try:
                # Pass 1: determine which column indices have at least one non-empty data value
                with open(path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    headers = next(reader, None)
                    if headers is None:
                        os.remove(path)
                        del file_paths[key]
                        continue

                    non_empty = set()
                    row_count = 0
                    for row in reader:
                        row_count += 1
                        for i, cell in enumerate(row):
                            if i < len(headers) and cell.strip():
                                non_empty.add(i)

                if row_count == 0:
                    os.remove(path)
                    del file_paths[key]
                    continue

                keep = sorted(non_empty)
                if len(keep) == len(headers):
                    continue  # all columns populated, nothing to rewrite

                # Pass 2: stream-rewrite keeping only non-empty columns
                tmp_path = path + '.tmp'
                try:
                    with open(path, 'r', encoding='utf-8') as f_in, \
                            open(tmp_path, 'w', newline='', encoding='utf-8') as f_out:
                        reader = csv.reader(f_in)
                        writer = csv.writer(f_out)
                        for row in reader:
                            writer.writerow([row[i] for i in keep if i < len(row)])
                    os.replace(tmp_path, path)
                except Exception:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    raise

            except FileNotFoundError:
                pass

    def flatten_dict(self, d, parent_key='', sep='.'):
        """
        Flatten nested dictionaries into a flat key-value structure.

        :param d: dictionary to flatten
        :param parent_key: prefix for nested keys
        :param sep: separator used to join nested keys
        :return: flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    @staticmethod
    def bson_file_to_csv_incremental(bson_file: str, csv_path: str, interpolated: bool = True) -> list:
        """
        Convert BSON file directly to CSV with incremental processing (memory efficient).
        Processes records one at a time without loading entire file into memory.

        :param bson_file: path to BSON file
        :param csv_path: base path for output CSV files (without extension)
        :param interpolated: whether to fill missing values
        :return: list of generated CSV file paths
        """
        parser = DATAParser(csv_path=csv_path, interpolated=interpolated, bson_file=bson_file)
        return parser.json_to_csv()

    @staticmethod
    def map_key(origin, key):
        """
        Map raw field keys to channel labels if defined.

        :param origin: origin of the data record
        :param key: raw key string
        :return: mapped key if available, otherwise the original key
        """
        mapping = CHANNEL_LABEL_MAPPINGS.get(origin, {})
        return mapping.get(key, key)
