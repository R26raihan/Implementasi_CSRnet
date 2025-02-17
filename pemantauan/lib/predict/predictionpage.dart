import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:async';
import 'package:latlong2/latlong.dart';

class PredictionPage extends StatefulWidget {
  @override
  _PredictionPageState createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  Map<String, int> predictions = {'DPR': 0, 'Bundaran HI': 0, 'Monas': 0};
  Timer? _timer;
  bool _isLoading = false;

  final Map<String, LatLng> locations = {
    'Monas': LatLng(-6.175392, 106.827153),
    'DPR': LatLng(-6.1764,  106.7959),
    'Bundaran HI': LatLng(-6.193667, 106.823024),
  };

  @override
  void initState() {
    super.initState();
    _startAutoRefresh();
    fetchPredictions();
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  void _startAutoRefresh() {
    _timer = Timer.periodic(Duration(seconds: 5), (timer) {
      fetchPredictions();
    });
  }

  Future<void> fetchPredictions() async {
    setState(() {
      _isLoading = true;
    });
    try {
      final response = await http.get(Uri.parse('http://192.168.1.8:8000/get_predictions'));
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        setState(() {
          predictions['DPR'] = data['DPR']['predicted_count'] ?? 0;
          predictions['Bundaran HI'] = data['Bundaran HI']['predicted_count'] ?? 0;
          predictions['Monas'] = data['Monas']['predicted_count'] ?? 0;
        });
      } else {
        print('Failed to fetch predictions');
      }
    } catch (e) {
      print('Error fetching predictions: $e');
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  // Fungsi untuk menentukan warna berdasarkan level kerumunan
  Color _getColorByCrowdLevel(int crowd) {
    if (crowd <= 500) {
      return Colors.green; // Tidak ada kerumunan
    } else if (crowd <= 1500) {
      return Colors.orange; // Potensi kerumunan
    } else {
      return Colors.red; // Kerumunan padat
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          FlutterMap(
            options: MapOptions(
              center: LatLng(-6.1900, 106.8228),
              zoom: 15.0,
            ),
            children: [
              TileLayer(
                urlTemplate: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
                subdomains: ['a', 'b', 'c'],
              ),
              MarkerLayer(
                markers: locations.entries.map((entry) {
                  final location = entry.key;
                  final position = entry.value;
                  final crowd = predictions[location] ?? 0;
                  return Marker(
                    width: 100.0,
                    height: 100.0,
                    point: position,
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Container(
                          padding: EdgeInsets.all(4),
                          decoration: BoxDecoration(
                            color: Colors.white,
                            borderRadius: BorderRadius.circular(4),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.black26,
                                blurRadius: 2,
                                offset: Offset(0, 2),
                              ),
                            ],
                          ),
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Text(
                                location,
                                style: TextStyle(
                                  color: Colors.black,
                                  fontSize: 12,
                                  fontWeight: FontWeight.bold,
                                ),
                                textAlign: TextAlign.center,
                              ),
                              Text(
                                '$crowd orang',
                                style: TextStyle(
                                  color: _getColorByCrowdLevel(crowd),
                                  fontSize: 12,
                                  fontWeight: FontWeight.bold,
                                ),
                                textAlign: TextAlign.center,
                              ),
                            ],
                          ),
                        ),
                        SizedBox(height: 4),
                        Icon(
                          Icons.location_on,
                          size: 40,
                          color: _getColorByCrowdLevel(crowd),
                        ),
                      ],
                    ),
                  );
                }).toList(),
              ),
            ],
          ),
          if (_isLoading)
            Center(
              child: CircularProgressIndicator(),
            ),
          // Widget untuk menampilkan informasi kerumunan
          Positioned(
            bottom: 20,
            left: 20,
            right: 20,
            child: Card(
              elevation: 4,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(10),
              ),
              child: Padding(
                padding: EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Level Kerumunan',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: Colors.blueAccent,
                      ),
                    ),
                    SizedBox(height: 8),
                    Text(
                      'Kerumunan hari ini',
                      style: TextStyle(
                        fontSize: 14,
                        color: Colors.grey[600],
                      ),
                    ),
                    SizedBox(height: 8),
                    // Informasi Monas
                    _buildLocationInfo('Monas', predictions['Monas'] ?? 0),
                    SizedBox(height: 8),
                    // Informasi Bundaran HI
                    _buildLocationInfo('Bundaran HI', predictions['Bundaran HI'] ?? 0),
                    SizedBox(height: 8),
                    // Informasi DPR
                    _buildLocationInfo('DPR', predictions['DPR'] ?? 0),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  // Widget untuk menampilkan informasi lokasi
  Widget _buildLocationInfo(String location, int crowd) {
    return Row(
      children: [
        Icon(
          Icons.location_on,
          color: _getColorByCrowdLevel(crowd),
        ),
        SizedBox(width: 8),
        Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              location,
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
              ),
            ),
            Text(
              '$crowd Orang',
              style: TextStyle(
                fontSize: 14,
                color: _getColorByCrowdLevel(crowd),
              ),
            ),
          ],
        ),
      ],
    );
  }
}

void main() {
  runApp(MaterialApp(
    home: PredictionPage(),
    debugShowCheckedModeBanner: false,
  ));
}