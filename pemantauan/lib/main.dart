import 'package:flutter/material.dart';
import 'package:pemantauan/predict/predictionpage.dart';

void main() {
  runApp(MaterialApp(
    home: PredictionPage(),
    debugShowCheckedModeBanner: false, // Menonaktifkan label debug
  ));
}


class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: PredictionPage(), // Hapus title karena PredictionPage tidak menerimanya
    );
  }
}
