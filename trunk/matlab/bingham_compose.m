function B = bingham_compose(B1,B2)
%B = bingham_compose(B1,B2) -- Compose two S^3 Binghams: B = quaternion_mult(B1,B2).
%Note that this is an approximation, as the Bingham distribution is not closed under composition.

if B1.d ~= 4 || B2.d ~= 4
    fprintf('Error: bingham_compose() is only defined for S^3 Binghams\n');
    B = [];
    return
end

S1 = bingham_scatter(B1);
S2 = bingham_scatter(B2);

a11 = S1(1,1);
a12 = S1(1,2);
a13 = S1(1,3);
a14 = S1(1,4);
a22 = S1(2,2);
a23 = S1(2,3);
a24 = S1(2,4);
a33 = S1(3,3);
a34 = S1(3,4);
a44 = S1(4,4);

b11 = S2(1,1);
b12 = S2(1,2);
b13 = S2(1,3);
b14 = S2(1,4);
b22 = S2(2,2);
b23 = S2(2,3);
b24 = S2(2,4);
b33 = S2(3,3);
b34 = S2(3,4);
b44 = S2(4,4);

S = zeros(4,4);
S(1,1) = a11*b11 - 2*a12*b12 - 2*a13*b13 - 2*a14*b14 + a22*b22 + 2*a23*b23 + 2*a24*b24 + a33*b33 + 2*a34*b34 + a44*b44;
S(1,2) = a11*b12 + a12*b11 + a13*b14 - a14*b13 - a12*b22 - a22*b12 - a13*b23 - a23*b13 - a14*b24 - a24*b14 - a23*b24 + a24*b23 - a33*b34 + a34*b33 - a34*b44 + a44*b34;
S(1,3) = a11*b13 + a13*b11 - a12*b14 + a14*b12 - a12*b23 - a23*b12 - a13*b33 + a22*b24 - a24*b22 - a33*b13 - a14*b34 - a34*b14 + a23*b34 - a34*b23 + a24*b44 - a44*b24;
S(1,4) = a11*b14 + a12*b13 - a13*b12 + a14*b11 - a12*b24 - a24*b12 - a22*b23 + a23*b22 - a13*b34 - a34*b13 - a23*b33 + a33*b23 - a14*b44 - a24*b34 + a34*b24 - a44*b14;
S(2,2) = 2*a12*b12 + a11*b22 + a22*b11 + 2*a13*b24 - 2*a14*b23 + 2*a23*b14 - 2*a24*b13 - 2*a34*b34 + a33*b44 + a44*b33;
S(2,3) = a12*b13 + a13*b12 + a11*b23 + a23*b11 - a12*b24 + a14*b22 - a22*b14 + a24*b12 + a13*b34 - a14*b33 + a33*b14 - a34*b13 + a24*b34 + a34*b24 - a23*b44 - a44*b23;
S(2,4) = a12*b14 + a14*b12 + a11*b24 + a12*b23 - a13*b22 + a22*b13 - a23*b12 + a24*b11 - a14*b34 + a34*b14 + a13*b44 + a23*b34 - a24*b33 - a33*b24 + a34*b23 - a44*b13;
S(3,3) = 2*a13*b13 + 2*a14*b23 - 2*a23*b14 + a11*b33 + a33*b11 - 2*a12*b34 + 2*a34*b12 - 2*a24*b24 + a22*b44 + a44*b22;
S(3,4) = a13*b14 + a14*b13 - a13*b23 + a23*b13 + a14*b24 - a24*b14 + a11*b34 + a12*b33 - a33*b12 + a34*b11 + a23*b24 + a24*b23 - a12*b44 - a22*b34 - a34*b22 + a44*b12;
S(4,4) = 2*a14*b14 - 2*a13*b24 + 2*a24*b13 + 2*a12*b34 - 2*a23*b23 - 2*a34*b12 + a11*b44 + a22*b33 + a33*b22 + a44*b11;
S(2,1) = S(1,2);
S(3,1) = S(1,3);
S(4,1) = S(1,4);
S(3,2) = S(2,3);
S(4,2) = S(2,4);
S(4,3) = S(3,4);

B = bingham_fit_scatter(S);
