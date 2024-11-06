(define (domain robot-arm)
  (:requirements :typing :equality :strips)

  (:types
    wire robot location workspace
  )

  (:predicates
    (holding ?wire - wire)
    (arm-empty ?arm - robot)
    ;(available ?wire - wire)
    (on ?wire - wire ?space - workspace)
    (locked ?wire - wire ?loc - location)
    (inserted ?wire - wire ?loc - location)
    (move-right ?arm - robot)
    (move-left ?arm - robot)
    (move-forward ?arm - robot)
    (move-backward ?arm - robot)
    (move-home ?arm - robot)
    (is-arm2 ?arm - robot)
    (is-arm1 ?arm - robot)   
  )

  (:action pickup
    :parameters
      (?arm - robot
       ?wire - wire
       ?space - workspace)
    :precondition
      (and
        ;(available ?wire)
        (on ?wire ?space)
        (arm-empty ?arm)
      )
    :effect
      (and
        ;(not (available ?wire))
        (not (on ?space ?wire))
        (holding ?wire)
        (not (arm-empty ?arm))
      )
  )

  (:action putdown
    :parameters
      (?arm - robot
       ?wire - wire
       ?space - workspace)
    :precondition
      (and
        (holding ?wire)
        (is-arm1 ?arm)
      )
    :effect
      (and
        (on ?wire ?space)
        ;(available ?wire)
        (arm-empty ?arm)
        (not (holding ?wire))
      )
  )

  (:action lock
    :parameters
      (?arm - robot
       ?wire - wire
       ?loc - location)
    :precondition
      (and
        (inserted ?wire ?loc)
        (is-arm2 ?arm)
      )
    :effect
      (and
        (locked ?wire ?loc)
        ;(not (available ?wire))
        (arm-empty ?arm)
        (not(inserted ?wire ?loc))
      )
  )

  (:action insert
    :parameters
      (?arm - robot
       ?wire - wire
       ?loc - location)
    :precondition
    (and
      (holding ?wire)
      (is-arm1 ?arm)
    )
    :effect
      (and
        (inserted ?wire ?loc)
        (not (holding ?wire))
      )
  )

  (:action move-forward
    :parameters (?arm - robot)
    :precondition (and) 
    :effect (and (move-forward ?arm))
  )

  (:action move-backward
    :parameters (?arm - robot)
    :precondition (and) 
    :effect (and (move-backward ?arm))
  )

  (:action move-right
    :parameters (?arm - robot)
    :precondition (and) 
    :effect (and (move-right ?arm))
  )

  (:action move-left
    :parameters (?arm - robot)
    :precondition (and) 
    :effect (and (move-left ?arm))
  )

  (:action move-home
    :parameters (?arm - robot)
    :precondition (and) 
    :effect (and (move-home ?arm))
  )
  
)