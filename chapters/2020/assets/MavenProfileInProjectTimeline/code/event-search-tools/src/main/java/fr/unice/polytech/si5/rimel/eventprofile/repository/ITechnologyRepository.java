package fr.unice.polytech.si5.rimel.eventprofile.repository;

import fr.unice.polytech.si5.rimel.eventprofile.domain.StackType;
import fr.unice.polytech.si5.rimel.eventprofile.model.Technology;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface ITechnologyRepository extends Neo4jRepository<Technology, String> {

    Optional<Technology> findByStack(StackType stackType);
}
