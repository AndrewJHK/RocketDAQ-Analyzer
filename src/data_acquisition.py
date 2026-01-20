import csv
import os
from datetime import datetime
from pathlib import Path
from pymongo import MongoClient, ASCENDING
from bson import decode_file_iter, json_util
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
    100: {
        "adc1.chan0.scaled": "TM2.scaled",
        "adc1.chan0.raw": "TM2.raw",
        "adc1.chan1.scaled": "PT2.scaled",
        "adc1.chan1.raw": "PT2.raw",
        "adc1.chan2.scaled": "PT1.scaled",
        "adc1.chan2.raw": "PT1.raw",
        "adc1.chan3.scaled": "TM1.scaled",
        "adc1.chan3.raw": "TM1.raw",
        "adc2.chan0.scaled": "PT5,scaled",
        "adc2.chan0.raw": "PT5,raw",
        "adc2.chan1.scaled": "PT6.scaled",
        "adc2.chan1.raw": "PT6.raw",
        "adc2.chan2.scaled": "PT4.scaled",
        "adc2.chan2.raw": "PT4.raw",
        "adc2.chan3.scaled": "PT3.scaled",
        "adc2.chan3.raw": "PT3.raw"

    }
}


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
    Parser for JSON/BSON data into CSV with interpolation and multi-device support.
    """

    def __init__(self, json_data, csv_path, interpolated):
        """
        Initialize the parser from JSON data.

        :param json_data: list of JSON documents loaded from BSON or JSON
        :param csv_path: path where CSV files will be written
        :param interpolated: whether to fill missing values (True) or leave them empty (False)
        """
        self.json_data = json_data
        self.csv_path = csv_path
        self.interpolated = interpolated
        self.fields_per_origin = {}
        self.last_known = {}
        self.counters = {}
        self.devices = {}

        self._extract_all_origins()
        self._initialize_devices()

    def _extract_all_origins(self):
        """
        Extract all origins and their available fields from JSON data.
        """
        for record in self.json_data:
            origin = record.get("data").get("header", {}).get("origin")
            if origin is None:
                continue
            if origin not in self.fields_per_origin:
                self.fields_per_origin[origin] = set()
            flat = self.flatten_dict(record.get("data").get("data", {}))
            for full_key in flat:
                mapped_key = self.map_key(origin, full_key)
                self.fields_per_origin[origin].add(f"data.{mapped_key}")

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

    def write_row(self, record, device: DeviceCSVConfig, writer):
        """
        Write a single JSON record as a row in a CSV file.

        :param record: single JSON document with header and data
        :param device: DeviceCSVConfig object for this origin
        :param writer: csv.DictWriter used for writing rows
        """
        origin = record["data"].get("header").get("origin", 0)
        timestamp_epoch_miliseconds = int(record["data"].get("header").get("timestamp", "1000190760000"))
        seconds = timestamp_epoch_miliseconds // 1000
        try:
            base_timestamp = datetime.fromtimestamp(seconds)
            timestamp_human = f"{base_timestamp.strftime('%Y-%m-%d %H:%M:%S')}.{timestamp_epoch_miliseconds}"
        except (OSError, ValueError):
            timestamp_human = "2001-09-11 08:46:00.000"

        row_data = {
            "header.origin": origin,
            "header.timestamp_epoch": timestamp_epoch_miliseconds,
            "header.timestamp_human": timestamp_human,
            "header.counter": self.counters[origin]
        }

        flattened = self.flatten_dict(record.get("data").get("data", {}))
        for field_key, value in flattened.items():
            mapped_key = self.map_key(origin, field_key)
            full_key = f"data.{mapped_key}"
            self.last_known[origin][full_key] = value
            row_data[full_key] = value

        for field in device.fieldnames:
            if field not in row_data:
                row_data[field] = self.last_known[origin].get(field) if self.interpolated else None

        non_header_keys = [k for k in row_data if not k.startswith("header") and k != "data.cpu_temperature"]
        if all(row_data.get(k) is None for k in non_header_keys):
            return

        writer.writerow(row_data)
        self.counters[origin] += 1

    def json_to_csv(self):
        """
        Export all JSON data into CSV files (one per origin).

        :return: list of paths to generated CSV files
        """
        sorted_data = sorted(self.json_data, key=self.get_timestamp)
        suffix = "_interpolated" if self.interpolated else "_none_filled"
        file_paths = {
            origin: f"{self.csv_path}{suffix}_{device.name}.csv"
            for origin, device in self.devices.items()
        }

        writers = {}
        files = {}
        try:
            for origin, device in self.devices.items():
                file = open(file_paths[origin], mode='w', newline='')
                writer = csv.DictWriter(file, fieldnames=device.fieldnames)
                writer.writeheader()
                self.last_known[origin] = {field: None for field in device.fieldnames}
                writers[origin] = writer
                files[origin] = file

            for record in sorted_data:
                origin = record["data"].get("header").get("origin")
                if origin in self.devices:
                    self.write_row(record, self.devices[origin], writers[origin])

        finally:
            for file in files.values():
                file.close()

        for key, path in list(file_paths.items()):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    reader = list(csv.reader(f))
                if len(reader) <= 1:
                    os.remove(path)
                    del file_paths[key]
                else:
                    # Drop empty columns
                    headers = reader[0]
                    transposed = list(zip(*reader[1:]))
                    non_empty_cols = [i for i, col in enumerate(transposed) if
                                      any(cell.strip() != '' for cell in col)]
                    cleaned = [[headers[i] for i in non_empty_cols]] + [
                        [row[i] for i in non_empty_cols] for row in reader[1:]
                    ]
                    with open(path, 'w', newline='', encoding='utf-8') as f_out:
                        writer = csv.writer(f_out)
                        writer.writerows(cleaned)
            except FileNotFoundError:
                pass
        return list(file_paths.values())

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
    def bson_files_to_json(bson_files: list[str]):
        for bson_file in bson_files:
            try:
                base, _ = os.path.splitext(bson_file)
                json_file = base + ".json"

                with open(bson_file, "rb") as f, open(json_file, "w", encoding="utf-8") as out:
                    out.write("[")
                    first = True
                    for doc in decode_file_iter(f):
                        if not first:
                            out.write(",")
                        first = False
                        out.write(json_util.dumps(doc, json_options=json_util.RELAXED_JSON_OPTIONS))
                    out.write("]")

                logger.info(f"Converted {bson_file} → {json_file}")
            except Exception as e:
                logger.error(f"Failed to convert {bson_file}: {e}")

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

    @staticmethod
    def get_timestamp(record):
        """
        Extract and return the timestamp from a record as an integer.

        :param record: JSON document with header.timestamp
        :return: timestamp as integer (epoch milliseconds)
        """
        timestamp_data = record["data"].get("header").get("timestamp", 1000190760000)
        return int(timestamp_data)
