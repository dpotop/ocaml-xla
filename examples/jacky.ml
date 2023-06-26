(* A very simple GPT implementation based on https://github.com/karpathy/nanoGPT
   This only contains the inference part as the xla crate does not support backpropagation.
   No dropout as this is inference only.

   This example requires the following tokenizer config file:
   https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe
   And the gpt2.npz weight file that can be extracted by running the get_nanogpt_weights.py script.
*)

open! Base
module Element_type = Xla.Element_type
module Literal = Xla.Literal
module Op = Xla.Op


let use_gpu = false

let temperature = 0.8
 
let () =

  let client = Xla.Client.cpu() in
  Stdio.printf "Platform name: %s\n" (Xla.Client.platform_name client);
  Stdio.printf "Platform version: %s\n%!" (Xla.Client.platform_version client);
  
  (* let root = Op.div (Op.r0_f32 1.0 ~builder) (Op.r0_f32 2.0 ~builder) in *)
  

  let builder = Xla.Builder.create ~name:"exe1" in
  let exe1 =
    let root = Op.div (Op.r0_f32 1.0 ~builder) (Op.parameter "tokens" ~id:0 ~ty:F32 ~dims:[| |] ~builder) in
    let computation = Xla.Computation.build ~root in
    Xla.Executable.compile client computation

  and exe2 =
    (* let builder = Xla.Builder.create ~name:"exe2" in *)
    let root = Op.add (Op.r0_f32 1.0 ~builder) (Op.parameter "tokens" ~id:0 ~ty:F32 ~dims:[| |] ~builder) in
    let computation = Xla.Computation.build ~root in
    Xla.Executable.compile client computation

  and exe3 =
    (* For an XLA convolution, per https://www.tensorflow.org/xla/operation_semantics#conv_convolution:
     * - the input shape is as follows: batch*features*spatial_dims
     * - the kernel shape is as follows: output_features*input_features*spatial_dims.
     * If called with values that do not satisfy these requirements, compilation fails. *)
    let root = Op.convolution
                 (Op.parameter "input"  ~id:0 ~ty:F32 ~dims:[|1;3;256;256|] ~builder)
                 (Op.parameter "kernel" ~id:1 ~ty:F32 ~dims:[|5;3;2;2|] ~builder)
                 ~strides:[|1;1|]
    in
    let computation = Xla.Computation.build ~root in
    Xla.Executable.compile client computation
  in
  
  for i = 1 to 10 do
    (* Create some input values *)
    let ba = Bigarray.Genarray.create Float32 C_layout [| |] in
    Bigarray.Genarray.set ba [| |] (Float.of_int i)  ;

    (* Execute exe1 *)
    let out1 = Xla.Executable.execute exe1 [| Literal.of_bigarray ba |] in
    let out1_0 = Xla.Buffer.to_literal_sync out1.(0).(0) in

    (* Execute exe2 *)
    let out2 = Xla.Executable.execute exe2 [| out1_0 |] in
    let out2_0 = Xla.Buffer.to_literal_sync out2.(0).(0) in

    (* Extract and print the output *)
    let outbigarray = Literal.to_bigarray ~kind:Float32 out2_0 in
    let value:float = Bigarray.Genarray.get outbigarray [| |] in
    Stdio.printf "Value:%f\n" value
  done ;

  (* Execute exe3 *)
  let img  =
    let data =
      Bigarray.Array1.of_array
        Float32
        C_layout
        (Array.init
           (1*256*256*3)
           ~f:(fun (i:int) :float -> Float.of_int(i)/.100.)
        )
    in
    Literal.of_bigarray (Bigarray.reshape (Bigarray.genarray_of_array1 data) [|1;3;256;256|])
  and kernel =
    let data =
      Bigarray.Array1.of_array
        Float32
        C_layout
        (Array.init
           (5*3*2*2)
           ~f:(fun (i:int) :float -> Float.of_int(i)/.100.)
        )
    in
    Literal.of_bigarray (Bigarray.reshape (Bigarray.genarray_of_array1 data) [|5;3;2;2|])
  in
  let out3 = Xla.Executable.execute exe3 [| img; kernel |] in
  let out3_0 = Xla.Buffer.to_literal_sync out3.(0).(0) in
  let shape = Bigarray.Genarray.dims (Literal.to_bigarray ~kind:Float32 out3_0) in
  Stdio.printf "Output shape: %d [|%d;%d;%d;%d|]\n"
    (Array.length shape)
    shape.(0)
    shape.(1)
    shape.(2)
    shape.(3)
