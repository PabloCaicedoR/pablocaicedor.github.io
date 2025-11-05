# Diccionario de datos

- **Knee_Flex_deg** (deg): Ángulo de flexión de rodilla (variable objetivo).
- **EMG_Quad_RMS_mV** (mV): RMS del EMG del cuádriceps (rectificado y promediado).
- **EMG_Ham_RMS_mV** (mV): RMS del EMG de isquiotibiales.
- **GRF_Vert_Norm_BW** (BW): Componente vertical de la fuerza de reacción del suelo, normalizada por peso corporal.
- **Omega_Shank_deg_s** (deg/s): Velocidad angular del segmento pierna (IMU).
- **Hip_Flex_deg** (deg): Ángulo de flexión de cadera.

## Generación (función subyacente)

	a = 15 
	+ 25 * sigmoid(1.6*(GRF - 0.6)) 
	+ 0.06 * sign(ω) * |ω|^1.2 
	+ 0.55 * Hip + 9*sin(Hip°) 
	- 10 * (EMG_ham / (EMG_quad + 0.05)) 
	+ 6 * EMG_quad * GRF
	- 40 * (EMG_quad - 0.12)^2
	+ 8 * GRF * sin(Hip°)
	+ ε

donde ε ~ Normal(0, 2) + 0.5*t(5). Se recortó a [-5, 140] deg.
