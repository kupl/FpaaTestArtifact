OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0), q(1), q(2)]
qreg q[3];


h q[0];
cx q[0],q[1];
cx q[0],q[2];
