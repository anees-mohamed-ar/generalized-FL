import tenseal as ts

def encrypt_and_add():
    # Ask for the first input
    first_input = float(input("Enter the first number: "))
    
    # Ask for the second input
    second_input = float(input("Enter the second number: "))

    # Set up TenSEAL context with CKKS encryption
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40  # Set the global scaling factor

    # Encrypt the first input
    encrypted_first_input = ts.ckks_vector(context, [first_input])
    encrypted_second_input = ts.ckks_vector(context, [second_input])

    # Perform addition: Add second input to the encrypted first input
    encrypted_result += encrypted_first_input + second_input


    # Decrypt the result
    decrypted_result = encrypted_result.decrypt()
    decrypted_first_input = encrypted_first_input.decrypt()
    decrypted_second_input = encrypted_second_input.decrypt()

    # Print all the decrypted values
    print("\nDecrypted First Input:", decrypted_first_input[0])
    print("Decrypted Second Input:", decrypted_second_input[0])
    print("Decrypted Result After Addition:", decrypted_result[0])

# Call the function
encrypt_and_add()
